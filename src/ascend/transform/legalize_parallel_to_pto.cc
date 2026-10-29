/*!
 * \file legalize_parallel_to_pto.cc
 * \brief LegalizeParallelToPTO: turn lane-varying control flow and the
 * remaining expression forms into value choices, so the vectorizer only sees
 * `BufferStore`s whose value is a `Select` tree.
 *
 * Pipeline position: `VerifyParallelToPTO -> LegalizeParallelToPTO ->
 * VectorizeParallelToPTO` (target-gated on PTO).
 *
 * The conditional rewrite runs inside the post-order statement visitor and has
 * two visible steps:
 *
 *  1. `NormalizeBranches` -- shape only. It collects both arms as flat store
 *     lists (inner conditionals are already folded into their values by the
 *     recursion) and splits them into canonical branches: a same-destination
 *     pair per matched store, one-sided branches for leftovers, and one branch
 *     per statement when an arm's stores depend on their own write order.
 *  2. `FoldCanonicalBranches` -- value only. Every canonical branch is exactly
 *     A1 (pair), A2 (then only) or A3 (else only), so this step builds the
 *     `Select` (A2/A3 read the destination's old value back) without ever
 *     searching for a partner.
 *
 * Min/Max become comparisons + `Select` and the tl/tir.infinity() sentinel
 * becomes a FloatImm constant in the same pass.
 */

#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <limits>
#include <set>
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

class ControlFlowMutator : public StmtExprMutator {
public:
  Stmt VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint != pto::kSimdVfBlockName) {
      // Enclosing blocks (root/tilelang_root/VECTOR/...) are traversed so the
      // SIMD_VF region is reached, but `in_simd_vf_` stays false: nothing
      // outside a converting region is rewritten.
      return StmtExprMutator::VisitStmt_(op);
    }
    // Any SIMD_VF block is traversed; no region-level predicate is needed.
    // Every rewrite below is gated on `in_parallel_`, which only becomes true
    // inside a `T.Parallel` loop, so a hand-written VMI region (a SIMD_VF
    // block without a unit) is left untouched by construction.
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

  // Post-order: an arm is rewritten first, so a nested `if` has already become a
  // store when the enclosing one is turned into a value choice.
  Stmt VisitStmt_(const IfThenElseNode *op) final {
    if (!in_parallel_) {
      return StmtExprMutator::VisitStmt_(op);
    }
    // Post-order: both arms are store lists by now (inner conditionals already
    // folded into their values), so normalization sees real stores and the
    // folder only ever handles A1/A2/A3.
    Stmt then_case = VisitStmt(op->then_case);
    Optional<Stmt> else_case;
    if (op->else_case.defined()) {
      else_case = VisitStmt(op->else_case.value());
    }
    std::vector<BranchItem> canonical = NormalizeBranches(then_case, else_case);
    return FoldCanonicalBranches(op->condition, canonical);
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

  // One canonical branch: exactly one of A1 (pair), A2 (then only), A3 (else
  // only). NormalizeBranches produces these; FoldCanonicalBranches consumes them.
  struct BranchItem {
    const BufferStoreNode *then_store = nullptr;
    const BufferStoreNode *else_store = nullptr;
  };

  // ---------------------------------------------------------------------
  // Step 1 (shape): collect both arms as flat store lists and split them into
  // canonical branches. No value is built here -- no `Select`, no read-back.
  //
  //  - an arm whose stores depend on their own order (RAW/WAR): one branch per
  //    statement, then-side first, else-side after it (source order)
  //  - otherwise: a same-destination pair per matched store (then order),
  //    leftovers as one-sided branches
  // ---------------------------------------------------------------------
  std::vector<BranchItem> NormalizeBranches(const Stmt &then_case,
                                      const Optional<Stmt> &else_case) {
    std::vector<const BufferStoreNode *> then_stores;
    std::vector<const BufferStoreNode *> else_stores;
    if (!pto::CollectStores(then_case, &then_stores)) {
      Fail("If then-branch under T.Parallel must be BufferStore(s) or no-op "
           "(Select-rewritable); unsupported stmt shape (Verify admits the "
           "same shapes)");
    }
    if (else_case.defined() && !pto::CollectStores(else_case.value(), &else_stores)) {
      Fail("If else-branch under T.Parallel must be BufferStore(s) or no-op "
           "(Select-rewritable); unsupported stmt shape (Verify admits the "
           "same shapes)");
    }

    std::vector<BranchItem> items;
    // Positional pairing first: when both arms have the same length and store i
    // of one arm writes the same destination as store i of the other, emitting
    // pair by pair preserves *both* arms' order, so an intra-arm RAW/WAR cannot
    // be reordered and the two stores can still share one value choice.
    if (then_stores.size() == else_stores.size()) {
      bool positional = !then_stores.empty();
      for (size_t i = 0; positional && i < then_stores.size(); ++i) {
        positional = pto::SameAccess(then_stores[i], else_stores[i]);
      }
      if (positional) {
        for (size_t i = 0; i < then_stores.size(); ++i) {
          BranchItem item;
          item.then_store = then_stores[i];
          item.else_store = else_stores[i];
          items.push_back(item);
        }
        return items;
      }
    }
    const bool order_matters = pto::HasIntraArmDependency(then_stores) ||
                               pto::HasIntraArmDependency(else_stores);
    if (order_matters) {
      // Merging would move a `write -> read` pair, so everything stays apart.
      for (const BufferStoreNode *th : then_stores) {
        BranchItem item;
        item.then_store = th;
        items.push_back(item);
      }
      for (const BufferStoreNode *el : else_stores) {
        BranchItem item;
        item.else_store = el;
        items.push_back(item);
      }
      return items;
    }

    std::vector<bool> else_used(else_stores.size(), false);
    std::vector<const BufferStoreNode *> partner(then_stores.size(), nullptr);
    for (size_t i = 0; i < then_stores.size(); ++i) {
      for (size_t j = 0; j < else_stores.size(); ++j) {
        if (!else_used[j] && pto::SameAccess(then_stores[i], else_stores[j])) {
          partner[i] = else_stores[j];
          else_used[j] = true;
          break;
        }
      }
    }
    for (size_t i = 0; i < then_stores.size(); ++i) {
      BranchItem item;
      item.then_store = then_stores[i];
      item.else_store = partner[i];  // nullptr -> one-sided (A2)
      items.push_back(item);
    }
    for (size_t j = 0; j < else_stores.size(); ++j) {
      if (!else_used[j]) {
        BranchItem item;
        item.else_store = else_stores[j];  // else only (A3)
        items.push_back(item);
      }
    }
    return items;
  }

  // ---------------------------------------------------------------------
  // Step 2 (value): every canonical branch is A1/A2/A3, so this is the whole
  // folder -- no partner search, no `may_pair`.
  // ---------------------------------------------------------------------
  Stmt FoldCanonicalBranches(const PrimExpr &condition,
                             const std::vector<BranchItem> &items) {
    if (items.empty()) {
      return Evaluate(0);
    }
    PrimExpr cond = VisitExpr(condition);
    Array<Stmt> out_stores;
    for (const BranchItem &item : items) {
      const BufferStoreNode *store =
          item.then_store != nullptr ? item.then_store : item.else_store;
      const std::string lane_error = pto::LaneVaryingStoreError(
          store->buffer, store->indices, parallel_var_);
      if (!lane_error.empty()) {
        Fail(lane_error);
      }
      PrimExpr then_value;
      PrimExpr else_value;
      if (item.then_store != nullptr && item.else_store != nullptr) {
        // A1: same destination, both arms present -- one value choice.
        ICHECK(pto::SameAccess(item.then_store, item.else_store))
            << "[LegalizeParallelToPTO] internal error: a canonical pair must "
               "share one destination";
        then_value = VisitExpr(item.then_store->value);
        else_value = VisitExpr(item.else_store->value);
      } else if (item.then_store != nullptr) {
        // A2: then only -- read the old value back.
        then_value = VisitExpr(item.then_store->value);
        else_value = BufferLoad(store->buffer, store->indices);
      } else {
        // A3: else only.
        then_value = BufferLoad(store->buffer, store->indices);
        else_value = VisitExpr(item.else_store->value);
      }
      out_stores.push_back(BufferStore(
          store->buffer, pto::ExpandPredSelect(cond, then_value, else_value),
          store->indices));
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
