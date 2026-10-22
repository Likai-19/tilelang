// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file expand_pto_fragment_fill_copy.cc
 * \brief ExpandPtoFragmentFillCopy: rewrite T.fill / T.copy calls that touch
 * a local.fragment buffer inside a converting PTO SIMD_VF region into the
 * equivalent T.Parallel loops, before layout inference.
 *
 * The loop forms carry the region semantics the PTO conversion needs:
 * a fill iterates its destination region (min/extent), a copy iterates the
 * copy region and maps the source and destination indices independently,
 * and the unit conversion's chunk mask limits partial coverage to the
 * active lanes, so lanes outside the region keep the previous value. After
 * the expansion every fragment access is an ordinary parallel-unit access
 * — layout inference assigns it the buffer's entry/lane mapping, the slot
 * plan reserves the backing, and Vectorize emits the handle-based
 * vload/vstore. The tileop.fill/copy call itself never reaches the
 * scheduler, so a fragment buffer never generates a fill/copy task.
 *
 * Fill/copy calls that touch no fragment buffer are left untouched for the
 * ordinary tileop lowering.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <sstream>
#include <string>

#include "../../op/copy.h"
#include "../../op/fill.h"
#include "../../op/operator.h"
#include "../target_utils.h"
#include "parallel_to_pto_utils.h"

namespace tvm {
namespace tl {
namespace {

using namespace tirx;

bool TouchesFragmentBuffer(const Call &call) {
  bool touches = false;
  PostOrderVisit(call, [&](const ObjectRef &node) {
    if (const auto *load = node.as<BufferLoadNode>()) {
      if (load->buffer.scope() == "local.fragment") {
        touches = true;
      }
    }
  });
  return touches;
}

bool IsFillOrCopyOp(const CallNode *call, std::string *name) {
  const auto *op_node = call->op.as<OpNode>();
  if (op_node == nullptr) {
    return false;
  }
  // The frontend lowers T.copy inside a SimdVF region to the typed
  // ascend_copy form (T.region(src) -> T.region(dst)); both it and the
  // plain copy op parse to a Copy with src/dst ranges.
  if (op_node->name == "tl.tileop.fill" || op_node->name == "tl.tileop.copy" ||
      op_node->name == "tl.tileop.ascend_copy") {
    *name = op_node->name;
    return true;
  }
  return false;
}

// MakeSIMTLoop attaches the SIMT layout hints of the original tileop call
// (coalesced width / parallel loop layout). The PTO path derives its own
// loop layouts from the buffer accesses, so the expanded loops start
// annotation-free.
Stmt StripLoopAnnotations(const Stmt &stmt) {
  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    Array<Stmt> seq_out;
    bool changed = false;
    for (const Stmt &s : seq->seq) {
      Stmt ns = StripLoopAnnotations(s);
      changed = changed || !ns.same_as(s);
      seq_out.push_back(ns);
    }
    if (!changed) {
      return stmt;
    }
    return SeqStmt::Flatten(seq_out);
  }
  if (const auto *attr = stmt.as<AttrStmtNode>()) {
    Stmt body = StripLoopAnnotations(attr->body);
    if (body.same_as(attr->body)) {
      return stmt;
    }
    return AttrStmt(attr->node, attr->attr_key, attr->value, body);
  }
  if (const auto *loop = stmt.as<ForNode>()) {
    Stmt body = StripLoopAnnotations(loop->body);
    if (body.same_as(loop->body) && loop->annotations.empty()) {
      return stmt;
    }
    return For(loop->loop_var, loop->min, loop->extent, loop->kind, body);
  }
  return stmt;
}

// A copy whose source and destination share one allocation expands into
// parallel units that read and write the same elements. Unit execution
// order is not defined, so unless the two regions are provably disjoint
// the expansion is a data race (a chunk overwrites elements a later chunk
// still has to read). Both regions are known only here — the converted
// units lose the region structure, and m2 §5.2 rules out cross-logical-
// point dependence analysis downstream — so the check lives in this pass.
// Disjointness is a per-dimension box test and therefore needs constant
// bounds and matching ranks; anything else is conservatively rejected.
void CheckSelfCopyRegionsDisjoint(const CopyNode *copy) {
  if (!copy->src->data.same_as(copy->dst->data)) {
    return;
  }
  if (!copy->src.same_as(copy->dst)) {
    // Two distinct views of one allocation: the per-dimension box test
    // below compares region indices only, and those do not account for
    // the views' element offsets — it can neither prove nor disprove the
    // overlap. Fail closed.
    std::ostringstream oss;
    oss << "[ExpandPtoFragmentFillCopy] overlapping self-copy: `"
        << copy->src->name << "` and `" << copy->dst->name
        << "` are two views of one allocation; disjointness of their "
           "regions cannot be proven. Copy through an intermediate buffer "
           "or operate on whole buffers";
    LOG(FATAL) << oss.str();
  }
  bool disjoint = false;
  bool provable = copy->src_range.size() == copy->dst_range.size() &&
                  !copy->src_range.empty();
  if (provable) {
    for (size_t i = 0; i < copy->src_range.size(); i++) {
      const int64_t *s_min = as_const_int(copy->src_range[i]->min);
      const int64_t *s_extent = as_const_int(copy->src_range[i]->extent);
      const int64_t *d_min = as_const_int(copy->dst_range[i]->min);
      const int64_t *d_extent = as_const_int(copy->dst_range[i]->extent);
      if (s_min == nullptr || s_extent == nullptr || d_min == nullptr ||
          d_extent == nullptr) {
        provable = false;
        break;
      }
      if (*s_min >= *d_min + *d_extent || *d_min >= *s_min + *s_extent) {
        disjoint = true;
        break;
      }
    }
  }
  if (!provable || !disjoint) {
    std::ostringstream oss;
    oss << "[ExpandPtoFragmentFillCopy] overlapping self-copy: `"
        << copy->src->name << "` is both copy source and destination, and "
        << (provable ? "the regions overlap" : "disjointness cannot be proven")
        << "; the expanded parallel units would race on the shared elements"
        << " (use an intermediate buffer or disjoint subregions)";
    LOG(FATAL) << oss.str();
  }
}

class RegionExpander : public StmtExprMutator {
public:
  // Rewrites the fill/copy calls of one converting region in source
  // order. Nested blocks inside the region keep their structure; the
  // base mutator rebuilds the standard wrappers around any replacement.
  Stmt Expand(const Stmt &stmt) { return VisitStmt(stmt); }

private:
  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    std::string name;
    if (call == nullptr || !IsFillOrCopyOp(call, &name)) {
      return GetRef<Stmt>(op);
    }
    Call fill_copy = GetRef<Call>(call);
    if (!TouchesFragmentBuffer(fill_copy)) {
      return GetRef<Stmt>(op);
    }
    TileOperator tile_op = ParseOperator(fill_copy);
    ICHECK(tile_op.defined())
        << "[ExpandPtoFragmentFillCopy] internal error: " << name
        << " call has no registered TileOperator builder";
    arith::Analyzer analyzer;
    Stmt expanded;
    // MakeSIMTLoop lives on the node classes (FillNode/CopyNode); the
    // TileOperator handle's arrow yields the base node.
    if (auto fill = tile_op.as<Fill>()) {
      expanded = fill.value().get()->MakeSIMTLoop(&analyzer);
    } else if (auto copy = tile_op.as<Copy>()) {
      CheckSelfCopyRegionsDisjoint(copy.value().get());
      expanded = copy.value().get()->MakeSIMTLoop(&analyzer);
    } else {
      ICHECK(false) << "[ExpandPtoFragmentFillCopy] internal error: " << name
                    << " parsed as an unexpected operator type";
    }
    ICHECK(expanded.defined())
        << "[ExpandPtoFragmentFillCopy] internal error: " << name
        << " produced no loop form";
    return StripLoopAnnotations(expanded);
  }
};

class ExpandPtoFragmentFillCopyImpl : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc f) {
    ExpandPtoFragmentFillCopyImpl mutator;
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = mutator.VisitStmt(fptr->body);
    return GetRef<PrimFunc>(fptr);
  }

private:
  // Converting SIMD_VF blocks get their fill/copy expanded; everything
  // else (including hand-written VMI regions and their nested blocks) is
  // only traversed so other converting regions in the function are found.
  Stmt VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint == "SIMD_VF" &&
        pto::ScanPtoRegion(op->body).has_any_parallel) {
      Stmt expanded = RegionExpander().Expand(op->body);
      if (expanded.same_as(op->body)) {
        return GetRef<Stmt>(op);
      }
      return SBlock(op->iter_vars, op->reads, op->writes, op->name_hint,
                    expanded, op->init, op->alloc_buffers, op->match_buffers,
                    op->annotations);
    }
    return StmtExprMutator::VisitStmt_(op);
  }
};

} // namespace

namespace transform {

tvm::transform::Pass ExpandPtoFragmentFillCopy() {
  auto pass_func = [](PrimFunc f, const IRModule &m,
                      const tvm::transform::PassContext &ctx) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsPTO(target.value())) {
      return f;
    }
    return ExpandPtoFragmentFillCopyImpl::Substitute(std::move(f));
  };
  return tvm::tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.ExpandPtoFragmentFillCopy", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.ExpandPtoFragmentFillCopy",
                        ExpandPtoFragmentFillCopy);
}

} // namespace transform

} // namespace tl
} // namespace tvm
