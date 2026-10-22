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

// Predicated whole-entry expansion for an unaligned 1D fragment region
// [min, min + extent): the loop base aligns down to the lane width, the
// original range becomes a lane-varying predicate that Legalize lowers
// into the store mask, and read/write indices shift by one shared
// constant so every iteration addresses whole entries.
For BuildPredicatedFill(const Buffer &dst, int64_t region_min,
                        int64_t region_extent, const PrimExpr &value,
                        int64_t lanes) {
  const DataType ity = DataType::Int(32);
  const int64_t delta = region_min % lanes;
  const int64_t base = region_min - delta;
  const int64_t padded =
      ((delta + region_extent + lanes - 1) / lanes) * lanes;
  Var i("i", ity);
  PrimExpr idx = IntImm(ity, static_cast<int>(base)) + i;
  PrimExpr cond =
      (idx >= IntImm(ity, static_cast<int>(region_min))) &&
      (idx < IntImm(ity, static_cast<int>(region_min + region_extent)));
  Stmt body =
      IfThenElse(cond, BufferStore(dst, value, {idx}));
  return For(i, IntImm(ity, 0), IntImm(ity, static_cast<int>(padded)),
             ForKind::kParallel, body);
}

For BuildPredicatedCopy(const Buffer &src, int64_t src_min, const Buffer &dst,
                        int64_t dst_min, int64_t extent, int64_t lanes) {
  const DataType ity = DataType::Int(32);
  const int64_t delta = dst_min % lanes;
  const int64_t base = dst_min - delta;
  const int64_t padded = ((delta + extent + lanes - 1) / lanes) * lanes;
  Var i("i", ity);
  PrimExpr dst_idx = IntImm(ity, static_cast<int>(base)) + i;
  PrimExpr src_idx =
      IntImm(ity, static_cast<int>(src_min - delta)) + i;
  PrimExpr cond =
      (dst_idx >= IntImm(ity, static_cast<int>(dst_min))) &&
      (dst_idx < IntImm(ity, static_cast<int>(dst_min + extent)));
  Stmt body = IfThenElse(cond, BufferStore(
                                   dst, BufferLoad(src, {src_idx}),
                                   {dst_idx}));
  return For(i, IntImm(ity, 0), IntImm(ity, static_cast<int>(padded)),
             ForKind::kParallel, body);
}

class RegionExpander : public StmtExprMutator {
public:
  // Rewrites the fill/copy calls of one converting region in source
  // order. Nested blocks inside the region keep their structure; the
  // base mutator rebuilds the standard wrappers around any replacement.
  // \p lanes is the region's vector width (tl.simdvf_lanes); it decides
  // whether a fragment region min is entry-aligned.
  RegionExpander(int64_t lanes) : lanes_(lanes) {}
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
    // The expansion expresses the fill/copy as plain element-wise
    // T.Parallel loops; call-level annotations (cluster/subblock routing
    // on ascend_copy, cache hints) describe semantics those loops do not
    // carry, so silently dropping them would change behavior. Reject.
    if (!fill_copy->annotations.empty()) {
      std::ostringstream keys;
      bool first = true;
      for (const auto &kv : fill_copy->annotations) {
        if (!first) {
          keys << ", ";
        }
        first = false;
        keys << kv.first;
      }
      std::ostringstream oss;
      oss << "[ExpandPtoFragmentFillCopy] " << name << " on a fragment "
             "buffer carries annotations ("
          << keys.str()
          << ") that the parallel-loop expansion cannot preserve; "
             "initialize through a T.Parallel loop instead";
      LOG(FATAL) << oss.str();
    }
    TileOperator tile_op = ParseOperator(fill_copy);
    ICHECK(tile_op.defined())
        << "[ExpandPtoFragmentFillCopy] internal error: " << name
        << " call has no registered TileOperator builder";
    // Unaligned fragment regions (min not a multiple of the lane width)
    // expand to the predicated whole-entry form; aligned ones keep the
    // plain MakeSIMTLoop shape. Both keep every access on whole entries
    // so the storage lane equals the execution lane.
    if (auto fill = tile_op.as<Fill>()) {
      const FillNode *node = fill.value().get();
      if (Stmt unaligned = TryUnalignedFill(node, name);
          unaligned.defined()) {
        return StripLoopAnnotations(unaligned);
      }
      arith::Analyzer analyzer;
      Stmt expanded = node->MakeSIMTLoop(&analyzer);
      return StripLoopAnnotations(expanded);
    }
    if (auto copy = tile_op.as<Copy>()) {
      const CopyNode *node = copy.value().get();
      CheckSelfCopyRegionsDisjoint(node);
      if (Stmt unaligned = TryUnalignedCopy(node, name);
          unaligned.defined()) {
        return StripLoopAnnotations(unaligned);
      }
      arith::Analyzer analyzer;
      Stmt expanded = node->MakeSIMTLoop(&analyzer);
      return StripLoopAnnotations(expanded);
    }
    ICHECK(false) << "[ExpandPtoFragmentFillCopy] internal error: " << name
                  << " parsed as an unexpected operator type";
    return Stmt();
  }

  // Predicated expansion of an unaligned 1D fill into a fragment. Any
  // other shape (aligned, multi-dimensional, non-constant bounds)
  // returns an undefined Stmt so the caller falls back to MakeSIMTLoop
  // (whose aligned form Verify proves) — multi-dimensional unaligned
  // regions are rejected here, closest to the access.
  Stmt TryUnalignedFill(const FillNode *node, const std::string &name) {
    if (node->region.size() != 1u || node->dst->shape.size() != 1u) {
      if (!AllMinsAligned(node->region)) {
        RejectUnalignedShape(name, node->dst);
      }
      return Stmt();
    }
    const int64_t *min_p = as_const_int(node->region[0]->min);
    const int64_t *extent_p = as_const_int(node->region[0]->extent);
    if (min_p == nullptr || extent_p == nullptr) {
      return Stmt();
    }
    if (*min_p % lanes_ == 0) {
      return Stmt();  // aligned: plain loop, lane proof admits it
    }
    return BuildPredicatedFill(node->dst, *min_p, *extent_p, node->value,
                               lanes_);
  }

  // Predicated expansion of an unaligned 1D copy touching a fragment.
  // Rules (design §4.3): a fragment destination whose min is unaligned
  // is only expandable when the source is the same allocation with a
  // congruent min (fragment->fragment) — the shared shift keeps both
  // sides in range. UB↔fragment copies with an unaligned fragment side
  // would read past the source range and are rejected.
  Stmt TryUnalignedCopy(const CopyNode *node, const std::string &name) {
    const bool dst_frag = node->dst.scope() == "local.fragment";
    const bool src_frag = node->src.scope() == "local.fragment";
    if (node->dst_range.size() != 1u || node->src_range.size() != 1u) {
      if (!AllMinsAligned(node->dst_range) ||
          (src_frag && !AllMinsAligned(node->src_range))) {
        RejectUnalignedShape(name, dst_frag ? node->dst : node->src);
      }
      return Stmt();
    }
    const int64_t *dmin_p = as_const_int(node->dst_range[0]->min);
    const int64_t *dext_p = as_const_int(node->dst_range[0]->extent);
    const int64_t *smin_p = as_const_int(node->src_range[0]->min);
    if (dmin_p == nullptr || dext_p == nullptr || smin_p == nullptr) {
      return Stmt();
    }
    const bool dst_aligned = *dmin_p % lanes_ == 0;
    const bool src_aligned = !src_frag || *smin_p % lanes_ == 0;
    if (dst_aligned && src_aligned) {
      return Stmt();
    }
    if (!dst_frag && !src_frag) {
      return Stmt();  // plain UB copy: not this pass's concern
    }
    if (dst_frag && src_frag && *dmin_p % lanes_ == *smin_p % lanes_) {
      return BuildPredicatedCopy(node->src, *smin_p, node->dst, *dmin_p,
                                 *dext_p, lanes_);
    }
    std::ostringstream oss;
    oss << "[ExpandPtoFragmentFillCopy] " << name
        << " has an unaligned fragment region (min not a multiple of "
           "the lane width "
        << lanes_
        << ") that cannot be expanded to whole-entry addressing; align "
           "the region start to the lane width or copy through an "
           "intermediate buffer";
    LOG(FATAL) << oss.str();
    return Stmt();
  }

  bool AllMinsAligned(const Array<Range> &region) const {
    for (const Range &range : region) {
      const int64_t *min_p = as_const_int(range->min);
      if (min_p != nullptr && *min_p % lanes_ != 0) {
        return false;
      }
    }
    return true;
  }

  [[noreturn]] void RejectUnalignedShape(const std::string &name,
                                         const Buffer &buffer) {
    std::ostringstream oss;
    oss << "[ExpandPtoFragmentFillCopy] " << name
        << " on fragment buffer `" << buffer->name
        << "` uses a multi-dimensional region with an unaligned min; "
           "only 1D unaligned fragment regions are supported (align the "
           "start to the lane width)";
    LOG(FATAL) << oss.str();
  }

  int64_t lanes_;
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
      int64_t lanes = 0;
      if (auto anno = op->annotations.Get("tl.simdvf_lanes");
          anno.has_value()) {
        if (const auto *imm = anno.value().as<IntImmNode>()) {
          lanes = imm->value;
        }
      }
      Stmt expanded = RegionExpander(lanes).Expand(op->body);
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
