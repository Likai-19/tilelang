/*!
 * \file legalize_parallel_to_pto.cc
 * \brief LegalizeParallelToPTO: safe scalar rewrites for the control-flow
 * forms VerifyParallelToPTO admitted in a converting PTO SIMD_VF region.
 *
 * Pipeline position: `VerifyParallelToPTO -> LegalizeParallelToPTO ->
 * VectorizeParallelToPTO` (target-gated on PTO). The pass rewrites the
 * scalar parallel TIR and produces scalar parallel TIR: lane-varying
 * If/Then/Else become the Select form, Min/Max become comparisons +
 * Select, and the tl/tir.infinity() constant becomes FloatImm, so the
 * vectorizer only needs vcmp/vcmps + vsel. It never emits
 * VMI, never calls a reducer, and never turns a conditional load into an
 * unconditional access outside the caller's full-width/masked load contract
 * (the rewrite keeps the load inside a Select arm, pre-evaluated under the
 * input program's memory contract, exactly like the other admitted Select
 * forms). A shape whose condition semantics cannot be preserved by the
 * rewrite is reported instead of being silently widened.
 *
 * The lane-uniform store specialisation is intentionally NOT part of this
 * pass: Verify already rejects lane-uniform stores, and this pass fails
 * defensively on that shape instead of materialising a reducer.
 *
 * This pass descends only into converting SIMD_VF blocks (those containing
 * a `T.Parallel` unit — the same predicate Verify uses); non-converting
 * regions are returned unchanged, so their Min/Max keep their code shape.
 */

#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <limits>
#include <sstream>
#include <vector>

#include "parallel_to_pto_utils.h"
#include "support/check.h"

#include "../target_utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

[[noreturn]] void Fail(const std::string &msg) {
  LOG(FATAL) << "[LegalizeParallelToPTO] " << msg;
}

bool SameAccess(const BufferStoreNode *a, const BufferStoreNode *b) {
  if (!a->buffer->data.same_as(b->buffer->data)) {
    return false;
  }
  if (a->indices.size() != b->indices.size()) {
    return false;
  }
  for (size_t i = 0; i < a->indices.size(); ++i) {
    if (!ffi::StructuralEqual()(a->indices[i], b->indices[i])) {
      return false;
    }
  }
  return true;
}

// Collect a flat list of BufferStores from a stmt (single store or Seq of
// stores). Nested Seq allowed; anything that is neither store nor no-op aborts
// collection.
bool CollectStores(const Stmt &stmt,
                   std::vector<const BufferStoreNode *> *out) {
  if (const auto *store = stmt.as<BufferStoreNode>()) {
    out->push_back(store);
    return true;
  }
  if (pto::IsNoOpStmt(stmt)) {
    return true;
  }
  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    for (const Stmt &s : seq->seq) {
      if (!CollectStores(s, out)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

// True when \p call is one of the boolean mask-combine ops that
// VectorizeParallelToPTO lowers to the VMI op \p vmi_name ("mask_and" /
// "mask_or" / "mask_not"): the shared name table carries every admitted
// spelling (tirx./tir. bitwise_* and tir.And|Or|Not).
bool IsMaskLogicCall(const CallNode *call, const char *vmi_name) {
  if (const auto *op = call->op.as<OpNode>()) {
    const char *mapped = pto::MaskLogicVmiOpName(op->name);
    return mapped != nullptr && std::string(mapped) == vmi_name;
  }
  return false;
}

// Expand And/Or/Not in the Select condition into nested Selects:
//   select(a&b, t, f) => select(a, select(b, t, f), f)
//   select(a|b, t, f) => select(a, t, select(b, t, f))
//   select(~a, t, f)  => select(a, f, t)
// (Boolean *values* under T.Parallel stay mask logic and are converted by
// VectorizeParallelToPTO to vand/vor/vnot.)
PrimExpr ExpandPredSelect(const PrimExpr &cond, const PrimExpr &tval,
                          const PrimExpr &fval) {
  if (const auto *andn = cond.as<AndNode>()) {
    return ExpandPredSelect(andn->a, ExpandPredSelect(andn->b, tval, fval),
                            fval);
  }
  if (const auto *orn = cond.as<OrNode>()) {
    return ExpandPredSelect(orn->a, tval, ExpandPredSelect(orn->b, tval, fval));
  }
  if (const auto *notn = cond.as<NotNode>()) {
    return ExpandPredSelect(notn->a, fval, tval);
  }
  if (const auto *call = cond.as<CallNode>()) {
    if (IsMaskLogicCall(call, "mask_and") && call->args.size() == 2) {
      return ExpandPredSelect(
          call->args[0], ExpandPredSelect(call->args[1], tval, fval), fval);
    }
    if (IsMaskLogicCall(call, "mask_or") && call->args.size() == 2) {
      return ExpandPredSelect(call->args[0], tval,
                              ExpandPredSelect(call->args[1], tval, fval));
    }
    if (IsMaskLogicCall(call, "mask_not") && call->args.size() == 1) {
      return ExpandPredSelect(call->args[0], fval, tval);
    }
  }
  return Select(cond, tval, fval);
}

class ControlFlowMutator : public StmtExprMutator {
public:
  Stmt VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint != pto::kSimdVfBlockName) {
      // Enclosing blocks (root/tilelang_root/VECTOR/...) are traversed so the
      // SIMD_VF region is reached, but `in_simd_vf_` stays false: nothing
      // outside a converting region is rewritten.
      return StmtExprMutator::VisitStmt_(op);
    }
    // Only the converting regions are this pass's subject: a SIMD_VF block
    // that holds a T.Parallel unit. A SIMD_VF block without T.Parallel
    // (hand-written VMI) is returned unchanged, so its Min/Max keep their
    // code shape.
    if (!pto::RegionContainsParallel(op->body)) {
      return GetRef<Stmt>(op);
    }
    bool saved = in_simd_vf_;
    in_simd_vf_ = true;
    Stmt body = StmtExprMutator::VisitStmt(op->body);
    in_simd_vf_ = saved;
    if (body.same_as(op->body)) {
      return GetRef<Stmt>(op);
    }
    SBlock block = GetRef<SBlock>(op);
    block.CopyOnWrite()->body = body;
    return block;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    if (!in_simd_vf_ || op->kind != ForKind::kParallel) {
      return StmtExprMutator::VisitStmt_(op);
    }

    Var saved_var = parallel_var_;
    bool saved_in = in_parallel_;
    parallel_var_ = op->loop_var;
    in_parallel_ = true;

    Stmt body = StmtExprMutator::VisitStmt(op->body);

    For loop = GetRef<For>(op);
    if (!body.same_as(op->body)) {
      loop.CopyOnWrite()->body = body;
    }

    parallel_var_ = saved_var;
    in_parallel_ = saved_in;
    return loop;
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    if (!in_parallel_) {
      return StmtExprMutator::VisitStmt_(op);
    }
    Stmt then_case = VisitStmt(op->then_case);
    Optional<Stmt> else_case;
    if (op->else_case.defined()) {
      else_case = VisitStmt(op->else_case.value());
    }
    return RewriteParallelIf(op->condition, then_case, else_case);
  }

  // Min/Max -> Select so Vectorize only needs cmp + vsel. Rewritten only
  // inside a T.Parallel unit in a converting region: scalar expressions in
  // sibling serial code within the same SIMD_VF keep their Min/Max shape.
  PrimExpr VisitExpr_(const MinNode *op) final {
    if (!in_parallel_) {
      return StmtExprMutator::VisitExpr_(op);
    }
    PrimExpr a = VisitExpr(op->a);
    PrimExpr b = VisitExpr(op->b);
    return Select(a < b, a, b);
  }
  PrimExpr VisitExpr_(const MaxNode *op) final {
    if (!in_parallel_) {
      return StmtExprMutator::VisitExpr_(op);
    }
    PrimExpr a = VisitExpr(op->a);
    PrimExpr b = VisitExpr(op->b);
    return Select(a > b, a, b);
  }

  // The frontend infinity() sentinel is a compile-time +inf constant of its
  // dtype; rewrite it to FloatImm so Vectorize's constant handling covers it
  // without a frontend-op-name case.
  PrimExpr VisitExpr_(const CallNode *op) final {
    if (in_parallel_) {
      if (const auto *op_node = op->op.as<OpNode>()) {
        const std::string &name = op_node->name;
        if (name == "tl.infinity" || name == "tir.infinity") {
          return FloatImm(op->dtype,
                          std::numeric_limits<double>::infinity());
        }
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }

private:
  bool IndexDependsOnParallelVar(const Array<PrimExpr> &indices) const {
    if (!parallel_var_.defined()) {
      return false;
    }
    Var pv = parallel_var_;
    for (const PrimExpr &idx : indices) {
      if (tirx::UsesVar(idx, [pv](const VarNode *v) {
            return pv.same_as(GetRef<Var>(v));
          })) {
        return true;
      }
    }
    return false;
  }

  Stmt RewriteParallelIf(const PrimExpr &condition, const Stmt &then_case,
                         const Optional<Stmt> &else_case) {
    std::vector<const BufferStoreNode *> then_stores;
    std::vector<const BufferStoreNode *> else_stores;
    const bool then_ok = CollectStores(then_case, &then_stores);
    const bool else_defined = else_case.defined();
    const bool else_ok =
        !else_defined || CollectStores(else_case.value(), &else_stores);

    if (!then_ok) {
      Fail("If then-branch under T.Parallel must be BufferStore(s) or no-op "
           "(Select-rewritable); unsupported stmt shape (Verify admits the "
           "same shapes)");
    }
    if (else_defined && !else_ok) {
      Fail("If else-branch under T.Parallel must be BufferStore(s) or no-op "
           "(Select-rewritable); unsupported stmt shape (Verify admits the "
           "same shapes)");
    }

    const bool then_noop = then_stores.empty();
    const bool else_noop = !else_defined || else_stores.empty();
    if (then_noop && else_noop) {
      return Evaluate(0);
    }

    PrimExpr cond = VisitExpr(condition);
    Array<Stmt> out_stores;

    auto emit_select_store = [&](const BufferStoreNode *th,
                                 const BufferStoreNode *el) {
      PrimExpr then_value;
      PrimExpr else_value;
      Buffer buf;
      Array<PrimExpr> indices;
      if (th != nullptr && el != nullptr) {
        // The caller pairs by SameAccess, so both arms share one Select.
        then_value = th->value;
        else_value = el->value;
        buf = th->buffer;
        indices = th->indices;
      } else if (th != nullptr) {
        then_value = th->value;
        else_value = BufferLoad(th->buffer, th->indices);
        buf = th->buffer;
        indices = th->indices;
      } else {
        then_value = BufferLoad(el->buffer, el->indices);
        else_value = el->value;
        buf = el->buffer;
        indices = el->indices;
      }

      // Lane-uniform destinations are out of scope; Verify rejects them
      // and this pass fails defensively instead of materialising a
      // reducer.
      if (!IndexDependsOnParallelVar(indices)) {
        std::ostringstream oss;
        oss << "lane-uniform store into buffer `" << buf->name
            << "` under T.Parallel is not supported at this stage; make "
               "the store lane-varying or hoist the computation out of "
               "the SimdVF region";
        Fail(oss.str());
      }

      PrimExpr then_v = VisitExpr(then_value);
      PrimExpr else_v = VisitExpr(else_value);
      PrimExpr selected = ExpandPredSelect(cond, then_v, else_v);
      out_stores.push_back(BufferStore(buf, selected, indices));
    };

    if (then_noop) {
      for (const BufferStoreNode *el : else_stores) {
        emit_select_store(nullptr, el);
      }
    } else if (else_noop) {
      for (const BufferStoreNode *th : then_stores) {
        emit_select_store(th, nullptr);
      }
    } else {
      // Pair stores by written address (buffer + indices); a store with no
      // counterpart keeps the other arm's value as a load of the same
      // address — the same masked-update form a one-sided branch produces.
      std::vector<bool> else_used(else_stores.size(), false);
      for (const BufferStoreNode *th : then_stores) {
        const BufferStoreNode *matched = nullptr;
        for (size_t j = 0; j < else_stores.size(); ++j) {
          if (!else_used[j] && SameAccess(th, else_stores[j])) {
            matched = else_stores[j];
            else_used[j] = true;
            break;
          }
        }
        emit_select_store(th, matched);
      }
      for (size_t j = 0; j < else_stores.size(); ++j) {
        if (!else_used[j]) {
          emit_select_store(nullptr, else_stores[j]);
        }
      }
    }

    if (out_stores.empty()) {
      return Evaluate(0);
    }
    if (out_stores.size() == 1) {
      return out_stores[0];
    }
    return SeqStmt(out_stores);
  }

  bool in_simd_vf_ = false;
  bool in_parallel_ = false;
  Var parallel_var_;
};

PrimFunc LowerControlFlow(PrimFunc f) {
  auto *n = f.CopyOnWrite();
  n->body = ControlFlowMutator()(std::move(n->body));
  return f;
}

} // namespace

namespace transform {

tvm::transform::Pass LegalizeParallelToPTO() {
  auto pass_func = [](PrimFunc f, const IRModule &,
                      const tvm::transform::PassContext &) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsPTO(target.value())) {
      return f;
    }
    return LowerControlFlow(std::move(f));
  };
  return tvm::tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.LegalizeParallelToPTO", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LegalizeParallelToPTO",
                        LegalizeParallelToPTO);
}

} // namespace transform

} // namespace tl
} // namespace tvm
