// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file verify_parallel_to_pto.cc
 * \brief Read-only input-contract verification for the PTO Parallel
 * vectorize path. Runs on the PTO target only.
 *
 * The pass checks every SIMD_VF block containing T.Parallel, reduce or fill:
 * lanes validity, loop structure/ranges, lane fragment presence and
 * invertibility, per-access patterns (continuous / lane-uniform), the
 * value-expression whitelist, element dtype constraints and region-level
 * rules (no GM access, no mixed hand-written VMI, no threadIdx binding).
 * References are checked against a lexical definition environment:
 * params and buffer data pointers at entry; outer loop vars,
 * thread_extent bindings, flat binds and buffer allocations registered
 * at their statement position and restored on scope exit, so a unit
 * sees exactly what is defined where it appears. The first error
 * terminates compilation with a `[VerifyParallelToPTO]` diagnostic.
 *
 * Pure hand-written VMI regions without high-level operations and SIMT_VF
 * blocks are skipped entirely. Direct tile operations use the VF lanes
 * annotation independently of any neighboring Parallel unit.
 */

#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <functional>
#include <optional>
#include <set>
#include <sstream>
#include <unordered_set>
#include <vector>

#include "../../transform/common/attr.h"
#include "buffer_version.h"
#include "parallel_to_pto_utils.h"
#include "pto_indexed_memory.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;
namespace pto_analysis = ::tvm::tl::pto;

namespace {

class VerifyParallelToPTOImpl : public StmtExprVisitor {
public:
  void Verify(const PrimFunc &f) {
    // The definition environment starts at function entry with
    // the params and the buffer data pointers. Every other definition is
    // registered at its lexical position during the traversal below and
    // removed when its scope exits.
    for (const auto &p : f->params) {
      defs_.insert(p.get());
    }
    for (const auto &kv : f->buffer_map) {
      defs_.insert(kv.second->data.get());
    }
    CollectVolatileAllocs(f->body);
    VisitStmt(f->body);
  }

private:
  // Function-level pre-scan for AllocBuffer nodes carrying the
  // tirx.volatile annotation. Presence of the annotation marks the
  // allocation volatile (PTO codegen semantics:
  // `annotations.count(tirx::attr::kVolatile)`); a False value is NOT
  // interpreted as opt-out. The buffer *data Var* identity is recorded,
  // so any other Buffer view over the same data is covered as well.
  // Lexical visibility of the var stays with the existing definition
  // checks — this set only answers "is this data volatile".
  void CollectVolatileAllocs(const Stmt &body) {
    PostOrderVisit(body, [&](const ObjectRef &n) {
      if (const auto *alloc = n.as<AllocBufferNode>()) {
        if (alloc->annotations.count(tirx::attr::kVolatile)) {
          volatile_allocs_.insert(alloc->buffer->data.get());
        }
      }
    });
  }

private:
  [[noreturn]] void Fail(const std::string &category,
                         const std::string &message) {
    std::ostringstream oss;
    oss << "[VerifyParallelToPTO] " << category << ": " << message;
    // Diagnostics carry the SIMD_VF/Parallel position. The prefix format
    // above is unchanged; the position is appended so existing
    // category/detail matching keeps working.
    if (!unit_context_.empty()) {
      oss << " [at " << unit_context_ << "]";
    }
    LOG(FATAL) << oss.str();
  }

  // Region-level context — failures between/around units report
  // at least the region (and the lane count once it is known).
  void SetRegionContext() {
    std::ostringstream oss;
    oss << "region #" << current_region_;
    if (current_lanes_ > 0) {
      oss << " (lanes " << current_lanes_ << ")";
    }
    unit_context_ = oss.str();
  }

  // ------------------------------------------------------------------
  // Unified unit discovery — same rules as the vectorize pass:
  // sequential units are allowed; each unit is a 1D Parallel or a
  // two-layer adjacent Parallel nest (2D). A third parallel
  // layer, a Parallel under IfThenElse/While, and other unsupported
  // structures are rejected explicitly.
  // ------------------------------------------------------------------
  int64_t ReadLanes(const SBlockNode *block) {
    auto anno = block->annotations.Get("tl.simdvf_lanes");
    if (!anno.has_value()) {
      Fail("lanes", "SIMD_VF block with high-level operations must carry the "
                    "tl.simdvf_lanes annotation (written by "
                    "T.SimdVF(lanes=...))");
    }
    auto imm = anno.value().try_cast<IntImm>();
    if (!imm.has_value() || imm.value()->dtype != DataType::Int(64)) {
      Fail("lanes", "tl.simdvf_lanes must be an Int64 constant");
    }
    int64_t lanes = imm.value()->value;
    if (lanes != 64 && lanes != 128 && lanes != 256) {
      std::ostringstream oss;
      oss << "T.SimdVF lanes must be one of {64, 128, 256}, got " << lanes;
      Fail("lanes", oss.str());
    }
    return lanes;
  }

  // ------------------------------------------------------------------
  // Region-level checks: everything inside a converting SIMD_VF region
  // must obey the region rules, not just the parallel body.
  // ------------------------------------------------------------------
  class RegionVisitor : public StmtExprVisitor {
  public:
    RegionVisitor(VerifyParallelToPTOImpl *parent) : parent_(parent) {}

    void VisitStmt_(const EvaluateNode *op) final {
      if (const auto *call_node = op->value.as<CallNode>()) {
        Call call = GetRef<Call>(call_node);
        if (pto_analysis::IsPtoDirectReduce(call) ||
            pto_analysis::IsPtoFill(call)) {
          // Only this root call is a supported statement. Its arguments and
          // annotations must still reject any nested direct tile operations.
          StmtExprVisitor::VisitExpr_(call_node);
          return;
        }
      }
      StmtExprVisitor::VisitStmt_(op);
    }
    void VisitStmt_(const BufferStoreNode *op) final {
      CheckScope(op->buffer);
      // The base visitor skips BufferStore.predicate
      // (stmt_functor.cc:85); accesses inside it must reach the same
      // admission check (the field sits outside the converting Parallel
      // unit, so the unit-level predicate rule cannot cover it).
      if (op->predicate.defined()) {
        VisitExpr(op->predicate.value());
      }
      StmtExprVisitor::VisitStmt_(op);
    }
    void VisitExpr_(const BufferLoadNode *op) final {
      CheckScope(op->buffer);
      // Same gap for BufferLoad.predicate (expr_functor.cc:36);
      // indices stay with the base visitor.
      if (op->predicate.defined()) {
        VisitExpr(op->predicate.value());
      }
      StmtExprVisitor::VisitExpr_(op);
    }
    void VisitExpr_(const CallNode *op) final {
      Call call = GetRef<Call>(op);
      if (pto_analysis::IsPtoDirectReduce(call) ||
          pto_analysis::IsPtoFill(call)) {
        parent_->Fail("tileop position",
                      "direct reduce and fill/clear must appear as a direct "
                      "Evaluate(Call) statement; Bind values and other nested "
                      "expressions are not supported");
      }
      if (const auto *op_node = op->op.as<OpNode>()) {
        std::string name = op_node->name;
        if (name.rfind("tl.vmi.", 0) == 0) {
          parent_->Fail(
              "mixing",
              "hand-written tl.vmi.* calls cannot be mixed with T.Parallel "
              "inside a converting SIMD_VF block");
        }
      }
      StmtExprVisitor::VisitExpr_(op);
    }
    void VisitStmt_(const AttrStmtNode *op) final {
      if (op->attr_key == tirx::attr::thread_extent) {
        // Name the binding that triggered the rejection: a region can carry
        // several thread_extent scopes and the message must say which one.
        // The prefix says "thread bindings" — this branch also rejects
        // blockIdx bindings, not only threadIdx.
        std::ostringstream oss;
        oss << "thread bindings are not allowed inside a converting "
               "SIMD_VF block (found `thread_extent` on binding `";
        if (const auto *iv = op->node.as<IterVarNode>()) {
          oss << iv->var->name_hint;
        } else {
          oss << op->node;
        }
        oss << "`)";
        parent_->Fail("region", oss.str());
      }
      // The legacy volatile_scope marker declares volatile access
      // semantics that are not currently supported; reject it under the
      // same rule instead of letting it pass as ordinary metadata. Scope
      // note: only markers *inside a converting region* are covered here
      // — a volatile_scope wrapping a whole SIMD_VF block from outside
      // never reaches this visitor and stays accepted.
      if (op->attr_key == tl::attr::volatile_scope) {
        parent_->Fail("memory access",
                      "volatile_scope attribute inside a converting region "
                      "marks accesses volatile; volatile accesses are not "
                      "yet supported");
      }
      // The base visitor only descends into value/body. A
      // PrimExpr held by `node` (metadata) can itself contain buffer
      // accesses (a bare BufferLoad or a composite Add/Cast over one, a
      // view over the same volatile data Var); run the same admission
      // walk over the full expression so those loads reach CheckScope.
      // Non-PrimExpr markers keep the existing contract (the unit-level
      // metadata check rejects containers).
      if (auto ne = op->node.try_cast<ObjectRef>()) {
        if (const auto *pe = ne.value().as<PrimExprNode>()) {
          VisitExpr(GetRef<PrimExpr>(pe));
        }
      }
      StmtExprVisitor::VisitStmt_(op);
    }
    void VisitStmt_(const ForNode *op) final {
      // A For-form thread binding (the public T.thread_binding API
      // constructs exactly this shape) must obey the same region-wide ban
      // as the thread_extent AttrStmt form above — the shared unit scan
      // only rejects wrappers *around* a Parallel, not sibling bindings.
      // This visitor only runs on regions that contain a Parallel, so
      // hand-written VMI regions without one are unaffected. Vectorized
      // sibling loops are intentionally NOT rejected here (out of scope).
      if (op->kind == ForKind::kThreadBinding) {
        std::ostringstream oss;
        oss << "thread bindings are not allowed inside a converting "
               "SIMD_VF block (found ThreadBinding loop on `"
            << op->loop_var->name_hint << "`";
        if (op->thread_binding.defined()) {
          oss << " bound to " << op->thread_binding.value()->thread_tag;
        }
        oss << ")";
        parent_->Fail("region", oss.str());
      }
      StmtExprVisitor::VisitStmt_(op);
    }

  private:
    void CheckScope(const Buffer &buffer) {
      // Volatile accesses are outside the supported contract
      // (an unused volatile read must not be DCE-deleted, and a live
      // uniform volatile read must not change its read count through the
      // sload+broadcast lowering). The data-Var identity is matched, so
      // any Buffer view over a volatile allocation is covered.
      if (parent_->volatile_allocs_.count(buffer->data.get()) != 0) {
        std::ostringstream oss;
        oss << "buffer `" << buffer->name
            << "` is allocated with the tirx.volatile annotation; volatile "
               "accesses are not yet supported";
        parent_->Fail("memory access", oss.str());
      }
      const std::string &scope = buffer.scope();
      if (scope == "global") {
        std::ostringstream oss;
        oss << "SIMD_VF block must not access GM directly (buffer `"
            << buffer->name << "` has scope global); stage through UB first";
        parent_->Fail("memory access", oss.str());
      }
      // Only UB accesses are currently supported. local /
      // local.fragment / shared.l1 (L1) and other scopes are rejected.
      if (scope != "shared" && scope != "shared.dyn") {
        std::ostringstream oss;
        oss << "buffer `" << buffer->name << "` has scope `" << scope
            << "`; only UB (shared/shared.dyn) accesses are supported";
        parent_->Fail("memory access", oss.str());
      }
      if (!pto_analysis::IsSupportedElementDType(buffer->dtype)) {
        std::ostringstream oss;
        oss << "buffer `" << buffer->name << "` element dtype " << buffer->dtype
            << " is not supported by the unified element "
               "addressing (no sub-byte packed types, no "
               "pre-vectorized elements, no FP64, 8/16/32-bit widths only)";
        parent_->Fail("dtype", oss.str());
      }
    }
    VerifyParallelToPTOImpl *parent_;
  };

  // ------------------------------------------------------------------
  // Unit-level checks
  // ------------------------------------------------------------------
  // Run before the expression/statement whitelist: Legalize may add old-value
  // reads to conditional stores, or turn branches into Select. Classify the
  // original access at its execution position, without following a saved
  // load's Var back to its definition and changing that execution position.
  class ConditionalIndexedAccessVisitor : public StmtExprVisitor {
  public:
    ConditionalIndexedAccessVisitor(
        VerifyParallelToPTOImpl *parent, const Var &loop_var,
        const pto_analysis::PtoInverseMapping &mapping,
        const std::set<const VarNode *> &external_defs,
        const Map<Var, PrimExpr> &bind_env = {})
        : parent_(parent), loop_var_(loop_var), mapping_(mapping),
          bind_env_(bind_env) {
      for (const VarNode *var : external_defs) {
        defs_.insert(GetRef<Var>(var));
      }
      defs_.insert(loop_var);
      for (const auto &[var, value] : bind_env) {
        defs_.insert(var);
      }
      if (mapping.is_2d) {
        defs_.insert(mapping.select_inner ? mapping.outer_var
                                          : mapping.inner_var);
      }
    }

    void VisitStmt_(const IfThenElseNode *op) final {
      std::ostringstream condition;
      condition << op->condition;
      InContext("IfThenElse condition `" + condition.str() + "`",
                [&] { VisitExpr(op->condition); });
      auto visit_branch = [&](const Stmt &body, const char *branch) {
        Map<Var, PrimExpr> saved_env = bind_env_;
        VarSet saved_defs = defs_;
        InContext(std::string("IfThenElse ") + branch + " branch (condition `" +
                      condition.str() + "`)",
                  [&] { VisitStmt(body); });
        bind_env_ = std::move(saved_env);
        defs_ = std::move(saved_defs);
      };
      visit_branch(op->then_case, "then");
      if (op->else_case.defined()) {
        visit_branch(op->else_case.value(), "else");
      }
    }

    void VisitExpr_(const SelectNode *op) final {
      InContext("Select condition", [&] { VisitExpr(op->condition); });
      InContext("Select true value", [&] { VisitExpr(op->true_value); });
      InContext("Select false value", [&] { VisitExpr(op->false_value); });
    }

    void VisitExpr_(const CallNode *op) final {
      if (auto call_op = op->op.as<Op>()) {
        const std::string &name = call_op.value()->name;
        if (name == "tirx.fabs" || name == "tirx.exp" || name == "tirx.log" ||
            name == "tirx.sqrt") {
          StmtExprVisitor::VisitExpr_(op);
        }
      }
      // Unknown calls (including unsupported conditional intrinsics) keep
      // their existing whitelist diagnostic; do not infer their evaluation.
    }

    void VisitExpr_(const LetNode *) final {}

    void VisitStmt_(const BindNode *op) final {
      InContextIfActive("Bind `" + std::string(op->var->name_hint) + "` value",
                        [&] { VisitExpr(op->value); });
      bind_env_.Set(op->var, op->value);
      defs_.insert(op->var);
    }

    void VisitStmt_(const AttrStmtNode *op) final {
      Map<Var, PrimExpr> saved_env = bind_env_;
      VarSet saved_defs = defs_;
      if (auto node = op->node.try_cast<PrimExpr>()) {
        VisitExpr(node.value());
      }
      StmtExprVisitor::VisitStmt_(op);
      bind_env_ = std::move(saved_env);
      defs_ = std::move(saved_defs);
    }

    // These forms are rejected by the existing unit whitelist. Do not infer
    // execution or lexical semantics for them in this narrower check.
    void VisitStmt_(const ForNode *) final {}
    void VisitStmt_(const WhileNode *) final {}
    void VisitStmt_(const SBlockNode *) final {}
    void VisitStmt_(const SBlockRealizeNode *) final {}
    void VisitStmt_(const AssertStmtNode *) final {}

    void VisitStmt_(const BufferStoreNode *op) final {
      VisitPredicatedAccess(op->predicate, "BufferStore", [&] {
        // Check the original store even when there is no old-value load.
        CheckAccess(op->buffer, op->indices, /*is_write=*/true);
        StmtExprVisitor::VisitStmt_(op);
      });
    }

    void VisitExpr_(const BufferLoadNode *op) final {
      VisitPredicatedAccess(op->predicate, "BufferLoad", [&] {
        // Nested index reads have their own access classification and may
        // require gather even when the enclosing payload access does not.
        StmtExprVisitor::VisitExpr_(op);
        CheckAccess(op->buffer, op->indices, /*is_write=*/false);
      });
    }

  private:
    using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;

    void InContext(const std::string &context,
                   const std::function<void()> &visit) {
      contexts_.push_back(context);
      visit();
      contexts_.pop_back();
    }

    void InContextIfActive(const std::string &context,
                           const std::function<void()> &visit) {
      if (contexts_.empty()) {
        visit();
      } else {
        InContext(context, visit);
      }
    }

    void VisitPredicatedAccess(const Optional<PrimExpr> &predicate,
                               const char *kind,
                               const std::function<void()> &visit) {
      if (!predicate.defined()) {
        visit();
        return;
      }
      const std::string context = std::string(kind) + " access predicate";
      // Base TIR visitors skip the predicate field. Inspect its reads even
      // if simplifying it would erase them, then check the guarded access.
      InContext(context + " condition", [&] { VisitExpr(predicate.value()); });
      if (is_one(analyzer_.Simplify(predicate.value()))) {
        visit();
      } else {
        InContext(context, visit);
      }
    }

    [[noreturn]] void FailAccess(const Buffer &buffer,
                                 const Array<PrimExpr> &indices, bool is_write,
                                 const std::string &reason) {
      std::ostringstream oss;
      oss << "buffer `" << buffer->name << "` " << (is_write ? "store" : "load")
          << " at " << BufferLoad(buffer, indices) << " " << reason << " in ";
      for (size_t i = 0; i < contexts_.size(); ++i) {
        if (i != 0)
          oss << " -> ";
        oss << contexts_[i];
      }
      oss << "; conditional gather/scatter is not yet supported";
      parent_->Fail("control flow", oss.str());
    }

    void CheckAccess(const Buffer &buffer, const Array<PrimExpr> &indices,
                     bool is_write) {
      if (contexts_.empty() ||
          !pto_analysis::IsOrdinaryUbScope(buffer.scope())) {
        return;
      }
      for (const PrimExpr &index : indices) {
        PostOrderVisit(index, [&](const ObjectRef &node) {
          if (auto var = node.as<Var>(); var && !defs_.count(var.value())) {
            FailAccess(buffer, indices, is_write,
                       "cannot be classified reliably: variable `" +
                           std::string(var.value()->name_hint) +
                           "` is not defined in this lexical scope");
          }
        });
      }
      pto_analysis::AccessPattern pattern;
      try {
        pattern = pto_analysis::AnalyzeBufferAccess(buffer, indices, is_write,
                                                    loop_var_, mapping_,
                                                    &analyzer_, bind_env_)
                      .pattern;
      } catch (const pto_analysis::PtoAnalysisError &err) {
        FailAccess(buffer, indices, is_write,
                   std::string("cannot be classified reliably: ") + err.what());
      }
      if (pattern == pto_analysis::AccessPattern::kIndexed ||
          (is_write && pattern == pto_analysis::AccessPattern::kLaneUniform)) {
        FailAccess(buffer, indices, is_write,
                   std::string(pto_analysis::AccessPatternToString(pattern)) +
                       " access requires " + (is_write ? "scatter" : "gather"));
      }
    }

    VerifyParallelToPTOImpl *parent_;
    Var loop_var_;
    const pto_analysis::PtoInverseMapping &mapping_;
    arith::Analyzer analyzer_;
    Map<Var, PrimExpr> bind_env_;
    VarSet defs_;
    std::vector<std::string> contexts_;
  };

  class UnitVisitor : public StmtExprVisitor {
  public:
    UnitVisitor(VerifyParallelToPTOImpl *parent, const Var &loop_var,
                const pto_analysis::PtoInverseMapping &mapping,
                const std::set<const VarNode *> &external_defs)
        : parent_(parent), loop_var_(loop_var), mapping_(mapping),
          external_defs_(external_defs) {
      // In a 2D unit `loop_var` is the vectorized coordinate; the
      // other one is kept as a serial loop and classifies as uniform.
      if (mapping.is_2d) {
        serial_var_ =
            mapping.select_inner ? mapping.outer_var : mapping.inner_var;
      }
    }

    void VisitStmt_(const AttrStmtNode *op) final {
      // Analyze the *whole* metadata expression with the shared
      // lane-use analysis — composite expressions (Add/Cast over loads)
      // classify correctly, bind identity is preserved, and unknown forms
      // produce a diagnostic instead of a crash. Containers are still
      // rejected.
      auto check_meta_expr = [&](const PrimExpr &meta) {
        // The unified reference check runs *first*, so a
        // ghost anywhere in the metadata expression (including load
        // indices, through any admitted wrapper node) reports the
        // definition diagnostic before address analysis can mis-route
        // it. The base class descends into every child expression, so
        // the earlier hand-written node table's gap (Div/Mod/Min/Max/
        // comparisons treated as leaves, loop-var references slipping
        // through them) cannot recur. The metadata position additionally
        // rejects direct loop-var references and vector-classified
        // binds: the metadata path only remaps scalar var references.
        RefCheck checker(this, /*metadata=*/true);
        checker(meta);
        // Metadata-mode classification (Sub/FloorMod/compare/
        // StringImm propagate instead of being value-whitelist rejects).
        pto_analysis::LaneUseContext ctx = MakeLaneUseContext();
        std::string why;
        pto_analysis::LaneUse use =
            pto_analysis::AnalyzeLaneUseMeta(meta, &ctx, &why);
        if (use == pto_analysis::LaneUse::kVarying) {
          parent_->Fail("expression",
                        "AttrStmt metadata depends on lane-varying values; "
                        "metadata cannot yet be given lane semantics");
        }
        if (use == pto_analysis::LaneUse::kUnknown) {
          std::ostringstream oss;
          oss << "AttrStmt metadata cannot be analyzed: " << why;
          parent_->Fail("expression", oss.str());
        }
      };
      // op->node may be an Any holding a PrimExpr; containers and other
      // object kinds are rejected.
      if (auto ne = op->node.try_cast<ObjectRef>()) {
        if (const auto *pe = ne.value().as<PrimExprNode>()) {
          check_meta_expr(GetRef<PrimExpr>(pe));
        } else {
          parent_->Fail("expression",
                        "AttrStmt metadata must be a PrimExpr (containers "
                        "cannot be given lane semantics)");
        }
      }
      // Non-object metadata (int/str atoms) carries no Var references.
      check_meta_expr(op->value);
      // An AttrStmt body is a nested statement scope — a Bind
      // defined inside it must not leak to later siblings, while outer
      // binds stay visible inside.
      Map<Var, PrimExpr> saved_env = bind_env_;
      Map<Var, Integer> saved_classified = classified_;
      StmtExprVisitor::VisitStmt_(op);
      bind_env_ = std::move(saved_env);
      classified_ = std::move(saved_classified);
    }
    // The StmtVisitor base class already dispatches several node
    // kinds (AllocBuffer/DeclBuffer/...) so VisitStmtDefault_ alone is
    // not a whitelist. Reject those kinds explicitly.

    void VisitStmt_(const ForNode *op) final {
      // No loops of any kind inside T.Parallel.
      parent_->Fail("loop structure",
                    "loops (serial/pipelined/while) inside T.Parallel are "
                    "not yet supported");
    }
    void VisitStmt_(const IfThenElseNode *) final {
      parent_->Fail("loop structure",
                    "conditional branches inside T.Parallel are not yet "
                    "supported");
    }
    void VisitStmt_(const WhileNode *) final {
      parent_->Fail("loop structure",
                    "while loops inside T.Parallel are not yet supported");
    }
    void VisitStmt_(const SBlockNode *) final {
      parent_->Fail("loop structure",
                    "nested blocks inside T.Parallel are not yet "
                    "supported");
    }
    void VisitStmt_(const SBlockRealizeNode *) final {
      parent_->Fail("loop structure",
                    "nested blocks inside T.Parallel are not yet "
                    "supported");
    }
    void VisitStmt_(const AllocBufferNode *) final {
      parent_->Fail("loop structure",
                    "buffer allocation inside T.Parallel is not yet "
                    "supported");
    }
    void VisitStmt_(const DeclBufferNode *) final {
      parent_->Fail("loop structure",
                    "buffer declaration inside T.Parallel is not yet "
                    "supported");
    }
    void VisitStmtDefault_(const ffi::Object *op) final {
      // Backstop for kinds the base does not dispatch.
      std::ostringstream oss;
      oss << "statement `" << op->GetTypeKey()
          << "` inside T.Parallel is not yet supported";
      parent_->Fail("loop structure", oss.str());
    }
    void VisitStmt_(const AssertStmtNode *) final {
      parent_->Fail("loop structure",
                    "assert statements inside T.Parallel are not yet "
                    "supported");
    }
    void VisitStmt_(const BindNode *op) final {
      // Self/forward reference is rejected before registering.
      bool self_ref = false;
      PostOrderVisit(op->value, [&](const ObjectRef &n) {
        if (auto *v = n.as<VarNode>()) {
          if (v == op->var.get()) {
            self_ref = true;
          }
        }
      });
      if (self_ref) {
        std::ostringstream oss;
        oss << "bind `" << op->var->name_hint
            << "` references itself in its definition";
        parent_->Fail("expression", oss.str());
      }
      // Bind right-hand sides run the shared
      // compute-value admission (whitelist, cast plans, op types,
      // references, lane classification) — the same entry as Evaluate
      // and store values.
      pto_analysis::LaneUse use = CheckComputeValue(
          op->value, "bind `" + op->var->name_hint + "` value");
      // Register only after the analysis succeeded.
      bind_env_.Set(op->var, op->value);
      classified_.Set(op->var,
                      Integer(use == pto_analysis::LaneUse::kUniform ? 0 : 1));
      StmtExprVisitor::VisitStmt_(op);
    }
    void VisitStmt_(const EvaluateNode *op) final {
      // An Evaluate value is a compute value — same admission as
      // Bind RHS and store values (it previously ran no lane
      // classification, so a StringImm reached Vectorize's internal
      // error).
      CheckComputeValue(op->value, "evaluate value");
      StmtExprVisitor::VisitStmt_(op);
    }
    void VisitStmt_(const BufferStoreNode *op) final {
      // The store value runs the shared compute-value
      // admission (whitelist, cast plans, op types, references, lane
      // classification).
      CheckComputeValue(op->value, "store value");
      // Store indices get the same unified reference check — the
      // address continuity proof can cancel a common ghost offset and
      // must not double as a definition check.
      CheckDefinedVars(op->indices);
      VerifyAccess(op->buffer, op->indices, /*is_write=*/true, op->predicate);
      StmtExprVisitor::VisitStmt_(op);
    }
    void VisitExpr_(const BufferLoadNode *op) final {
      // Load indices are checked against the same definition
      // environment as values and store indices.
      CheckDefinedVars(op->indices);
      VerifyAccess(op->buffer, op->indices, /*is_write=*/false, op->predicate);
      StmtExprVisitor::VisitExpr_(op);
    }

  private:
    // One construction point for the lane-use context. The
    // metadata, operation-type and bind-definition entries share the same
    // definition environment, classification cache and guards.
    pto_analysis::LaneUseContext MakeLaneUseContext() {
      pto_analysis::LaneUseContext ctx;
      ctx.loop_var = loop_var_;
      ctx.serial_var = serial_var_;
      ctx.mapping = mapping_;
      ctx.analyzer = &analyzer_;
      ctx.bind_env = bind_env_;
      ctx.external_defs = external_defs_;
      ctx.classified = classified_;
      return ctx;
    }

    // The unified reference check. One StmtExprVisitor
    // covers every expression position — values, load/store indices and
    // metadata node/value — relying on the base class's complete descent
    // (BufferLoad indices and every wrapper node the admission tables
    // accept are all visited). Every Var must be the loop var, a bind
    // defined earlier in this unit, or a definition in the lexical
    // snapshot; the metadata position additionally rejects direct
    // loop-var references and vector-classified binds because the
    // metadata path only remaps scalar var references.
    class RefCheck : public StmtExprVisitor {
    public:
      RefCheck(UnitVisitor *parent, bool metadata)
          : parent_(parent), metadata_(metadata) {}
      void VisitExpr_(const VarNode *v) final {
        if (parent_->serial_var_.defined() && v == parent_->serial_var_.get()) {
          // The kept coordinate becomes the new outer serial loop
          // variable. It is a uniform scalar and is not rewritten by the
          // metadata remap, so a direct metadata reference stays legal.
          return;
        }
        if (v == parent_->loop_var_.get()) {
          if (metadata_) {
            parent_->parent_->Fail(
                "expression",
                "AttrStmt metadata directly references the loop variable "
                "(even if algebraically uniform); the metadata path cannot "
                "rewrite it");
          }
          return;
        }
        if (parent_->bind_env_.count(GetRef<Var>(v))) {
          if (metadata_ && parent_->classified_.count(GetRef<Var>(v))) {
            Integer cls = parent_->classified_[GetRef<Var>(v)];
            if (cls->value != 0) {
              std::ostringstream oss;
              oss << "AttrStmt metadata references bind `" << v->name_hint
                  << "` whose value converts to a vector; metadata must "
                     "stay scalar";
              parent_->parent_->Fail("expression", oss.str());
            }
          }
          return;
        }
        if (parent_->external_defs_.count(v) != 0) {
          return;
        }
        std::ostringstream oss;
        oss << "variable `" << v->name_hint
            << "` is used but never defined in this kernel";
        parent_->parent_->Fail("expression", oss.str());
      }

    private:
      UnitVisitor *parent_;
      bool metadata_;
    };

    void CheckDefinedVars(const PrimExpr &expr) {
      RefCheck checker(this, /*metadata=*/false);
      checker(expr);
    }
    void CheckDefinedVars(const Array<PrimExpr> &indices) {
      RefCheck checker(this, /*metadata=*/false);
      for (const auto &idx : indices) {
        checker(idx);
      }
    }

    // One admission entry for every compute-value position (Bind
    // RHS, Evaluate, store value): whitelist, cast plans, operation
    // types, reference check and lane classification. Before this only
    // the Bind path ran the lane classification, so a value the
    // whitelist lets through but that is not a legal scalar computation
    // (e.g. an external vector Var) reached Vectorize through Evaluate or
    // a store. `what` is the diagnostic prefix for the position.
    pto_analysis::LaneUse CheckComputeValue(const PrimExpr &value,
                                            const std::string &what) {
      if (auto reason = pto_analysis::FindUnsupportedExprNode(value)) {
        parent_->Fail("expression", what + ": " + reason.value());
      }
      CheckValueCasts(value);
      CheckBinaryOpTypes(value);
      CheckDefinedVars(value);
      pto_analysis::LaneUseContext ctx = MakeLaneUseContext();
      std::string why;
      pto_analysis::LaneUse use =
          pto_analysis::AnalyzeLaneUse(value, &ctx, &why);
      if (use == pto_analysis::LaneUse::kUnknown) {
        std::ostringstream oss;
        oss << what << " cannot be classified: " << why;
        parent_->Fail("expression", oss.str());
      }
      return use;
    }

    /*! The two-operand value nodes, one shared recursion shape: both
     * operands run through the same check, and each kind additionally
     * applies the VMI operation matrix. `vmi_op` carries the kind's VMI op
     * (vadd/vsub/vmul/vdiv). The unary Not and the three-operand Select are
     * handled separately by the caller. */
    struct BinaryOperands {
      const char *vmi_op;
      DataType dtype;
      PrimExpr a;
      PrimExpr b;
    };

    static std::optional<BinaryOperands> GetBinaryOperands(
        const PrimExpr &e) {
      if (const auto *n = e.as<AddNode>()) {
        return BinaryOperands{"vadd", n->dtype, n->a, n->b};
      }
      if (const auto *n = e.as<SubNode>()) {
        return BinaryOperands{"vsub", n->dtype, n->a, n->b};
      }
      if (const auto *n = e.as<MulNode>()) {
        return BinaryOperands{"vmul", n->dtype, n->a, n->b};
      }
      if (const auto *n = e.as<DivNode>()) {
        return BinaryOperands{"vdiv", n->dtype, n->a, n->b};
      }
      return std::nullopt;
    }

    // Per-operation type rules driven by the operands'
    // lane classification. Uniform+Uniform stays scalar (only the base
    // dtype and PTO scalar-support are required); any Varying operand
    // makes the node a vector operation and the VMI op matrix applies.
    // Inner sub-expressions are classified individually, so an int8
    // scalar Mul inside a Varying Add is legal while an int8 vector Mul
    // stays rejected.
    void CheckBinaryOpTypes(const PrimExpr &expr, bool indexed_offset = false) {
      std::function<void(const PrimExpr &)> check = [&](const PrimExpr &e) {
        if (auto bin = GetBinaryOperands(e)) {
          check(bin->a);
          check(bin->b);
          CheckOneOp(bin->vmi_op, bin->dtype, bin->a, bin->b, indexed_offset);
          return;
        }
        if (const auto *cast = e.as<CastNode>()) {
          check(cast->value);
          return;
        }
        if (const auto *call = e.as<CallNode>()) {
          if (auto call_op = call->op.as<Op>()) {
            const std::string &name = call_op.value()->name;
            if (name == "tirx.fabs" || name == "tirx.exp" ||
                name == "tirx.log" || name == "tirx.sqrt") {
              check(call->args[0]);
              CheckOneUnaryOp(name.c_str(), call->dtype, call->args[0]);
              return;
            }
          }
        }
        if (const auto *load = e.as<BufferLoadNode>()) {
          // Index arithmetic is address analysis, not value arithmetic.
          (void)load;
          return;
        }
        // Vars/constants carry no operation to check.
      };
      check(expr);
    }

    void CheckOneUnaryOp(const char *op_name, DataType dtype,
                         const PrimExpr &arg) {
      pto_analysis::LaneUseContext ctx = MakeLaneUseContext();
      std::string why;
      pto_analysis::LaneUse a = pto_analysis::AnalyzeLaneUse(arg, &ctx, &why);
      if (a == pto_analysis::LaneUse::kUnknown) {
        return;
      }
      const bool vector_op = a == pto_analysis::LaneUse::kVarying;
      if (pto_analysis::IsSupportedFloat8(dtype)) {
        std::ostringstream oss;
        oss << op_name << " on "
            << (vector_op ? "lane-varying " : "lane-uniform ") << dtype
            << " is not supported (FP8 arithmetic is out of scope; convert "
               "to float32 explicitly first)";
        parent_->Fail("dtype", oss.str());
      }
      if (vector_op) {
        if (!pto_analysis::IsSupportedVectorUnaryOp(op_name, dtype)) {
          std::ostringstream oss;
          oss << op_name << " on " << dtype
              << " elements is not supported by the VMI operation matrix";
          parent_->Fail("dtype", oss.str());
        }
        return;
      }
      if (pto_analysis::IsFloatFamily(dtype) &&
          pto_analysis::ContainsRuntimeValue(arg)) {
        std::ostringstream oss;
        oss << op_name << " on " << dtype
            << " is a runtime lane-uniform float computation; it stays "
               "scalar inside SIMD_VF (no scalar float ALU by design). "
               "Compute it outside the SimdVF region, or wait for the "
               "gather/scatter milestone for vector write-back";
        parent_->Fail("dtype", oss.str());
      }
    }

    pto_analysis::LaneUse ClassifyComputeOperand(const PrimExpr &expr,
                                                 bool indexed_offset,
                                                 std::string *why) {
      if (indexed_offset) {
        if (pto_analysis::IsPtoScalarCoordinate(expr, loop_var_, bind_env_)) {
          return pto_analysis::LaneUse::kUniform;
        }
        // Address expressions additionally admit scalar chunk/kept-coordinate
        // selection. Use the same bounded, loaded-value-aware
        // proof as ConvertExpr instead of the narrower value whitelist.
        const PrimExpr *lhs = nullptr;
        const PrimExpr *rhs = nullptr;
        if (const auto *add = expr.as<AddNode>()) {
          lhs = &add->a;
          rhs = &add->b;
        } else if (const auto *sub = expr.as<SubNode>()) {
          lhs = &sub->a;
          rhs = &sub->b;
        } else if (const auto *mul = expr.as<MulNode>()) {
          lhs = &mul->a;
          rhs = &mul->b;
        } else if (const auto *div = expr.as<DivNode>()) {
          lhs = &div->a;
          rhs = &div->b;
        }
        if (lhs != nullptr) {
          pto_analysis::LaneUse a =
              ClassifyComputeOperand(*lhs, indexed_offset, why);
          pto_analysis::LaneUse b =
              ClassifyComputeOperand(*rhs, indexed_offset, why);
          if (a == pto_analysis::LaneUse::kUnknown ||
              b == pto_analysis::LaneUse::kUnknown) {
            return pto_analysis::LaneUse::kUnknown;
          }
          // ConvertBinary retains vector representation even when an
          // algebraic identity such as saved_index - saved_index is uniform.
          return a == pto_analysis::LaneUse::kUniform &&
                         b == pto_analysis::LaneUse::kUniform
                     ? pto_analysis::LaneUse::kUniform
                     : pto_analysis::LaneUse::kVarying;
        }
        if (const auto *cast = expr.as<CastNode>()) {
          return ClassifyComputeOperand(cast->value, indexed_offset, why);
        }
        if (expr.as<ModNode>() || expr.as<MinNode>() || expr.as<MaxNode>() ||
            expr.as<SelectNode>()) {
          Optional<Var> kept_var;
          int64_t kept_extent = 0;
          if (mapping_.is_2d) {
            kept_var = serial_var_;
            kept_extent = mapping_.select_inner ? mapping_.outer_extent
                                                : mapping_.inner_extent;
          }
          const bool uniform = pto_analysis::IsPtoLaneUniform(
              expr, loop_var_, mapping_.index_expr, mapping_.lane_var,
              mapping_.chunk_var, mapping_.lanes, mapping_.chunk_count,
              kept_var, kept_extent, bind_env_);
          if (uniform)
            return pto_analysis::LaneUse::kUniform;
          if (why != nullptr) {
            *why = "indexed coordinate requires unsupported lane-varying "
                   "division/modulo or selection";
          }
          return pto_analysis::LaneUse::kUnknown;
        }
      }
      pto_analysis::LaneUseContext ctx = MakeLaneUseContext();
      return pto_analysis::AnalyzeLaneUse(expr, &ctx, why);
    }

    void CheckOneOp(const char *op_name, DataType dtype, const PrimExpr &lhs,
                    const PrimExpr &rhs, bool indexed_offset) {
      std::string why;
      pto_analysis::LaneUse a =
          ClassifyComputeOperand(lhs, indexed_offset, &why);
      pto_analysis::LaneUse b =
          ClassifyComputeOperand(rhs, indexed_offset, &why);
      if (a == pto_analysis::LaneUse::kUnknown ||
          b == pto_analysis::LaneUse::kUnknown) {
        if (indexed_offset) {
          parent_->Fail("expression", std::string("indexed offset ") + op_name +
                                          " cannot be classified: " + why);
        }
        // The operand's access analysis failed (stride/pattern/etc.); the
        // dedicated BufferLoad visitor reports that with the correct
        // category later in this same unit walk. Do not double-report it
        // as a dtype error here.
        return;
      }
      bool vector_op = (a == pto_analysis::LaneUse::kVarying ||
                        b == pto_analysis::LaneUse::kVarying);
      // FP8 arithmetic is rejected in both forms, and deliberately before the
      // generic matrix: the conversion matrix admits FP8 operands, so "not in
      // the VMI op matrix" alone would be an opaque reason. No implicit
      // FP8 -> FP32 promotion is inserted either.
      if (pto_analysis::IsSupportedFloat8(dtype)) {
        std::ostringstream oss;
        oss << op_name << " on "
            << (vector_op ? "lane-varying " : "lane-uniform ") << dtype
            << " is not supported (FP8 arithmetic is not supported; convert "
               "to float32 explicitly first)";
        parent_->Fail("dtype", oss.str());
      }
      if (vector_op) {
        // The node emits a VMI op: apply the operation matrix.
        if (!pto_analysis::IsSupportedVectorBinaryOp(op_name, dtype)) {
          std::ostringstream oss;
          oss << op_name << " on " << dtype
              << " elements is not supported by the VMI operation matrix";
          parent_->Fail("dtype", oss.str());
        }
        return;
      }
      // Lane-uniform stays scalar by design, so a runtime-valued float
      // computation here runs on the scalar float ALU inside SIMD_VF.
      // Compile-time constants fold and integer scalars have no such
      // premise, so only float-family runtime values are rejected.
      if (pto_analysis::IsFloatFamily(dtype) &&
          (pto_analysis::ContainsRuntimeValue(lhs) ||
           pto_analysis::ContainsRuntimeValue(rhs))) {
        std::ostringstream oss;
        oss << op_name << " on " << dtype
            << " is a runtime lane-uniform float computation; it stays "
               "scalar inside SIMD_VF (no scalar float ALU by design). "
               "Compute it outside the SimdVF region, or wait for the "
               "gather/scatter milestone for vector write-back";
        parent_->Fail("dtype", oss.str());
      }
      // Pure scalar op: base dtype legality only (int8 scalar mul is
      // fine; int24/FP64 still rejected by the base dtype rule).
      if (!pto_analysis::IsSupportedElementDType(dtype)) {
        std::ostringstream oss;
        oss << "scalar " << op_name << " on " << dtype
            << " is not supported by the base element dtype rule";
        parent_->Fail("dtype", oss.str());
      }
    }

  public:
    // Unified value-Cast check: called for Bind, Evaluate
    // and store values alike. A Cast is only accepted when PlanValueCast
    // proves a VMI-equivalent conversion (attributes included); Cast
    // nodes inside BufferLoad *indices* are index arithmetic handled by
    // address analysis and are not checked here.
    void CheckValueCasts(const PrimExpr &expr, bool indexed_offset = false) {
      class CastCheck : public StmtExprVisitor {
      public:
        CastCheck(UnitVisitor *parent, bool indexed_offset)
            : parent_(parent), indexed_offset_(indexed_offset) {}
        void VisitExpr_(const CastNode *op) final {
          // Classify the *input of this Cast* with the existing lane-use
          // analysis, then hand the explicit kind to the shared planner.
          std::string lane_why;
          pto_analysis::LaneUse use = parent_->ClassifyComputeOperand(
              op->value, indexed_offset_, &lane_why);
          if (use == pto_analysis::LaneUse::kUnknown) {
            std::ostringstream oss;
            oss << "Cast " << op->value.dtype() << " -> " << op->dtype
                << " cannot classify its input: " << lane_why;
            parent_->parent_->Fail("dtype", oss.str());
            return;
          }
          pto_analysis::CastContext context;
          if (use == pto_analysis::LaneUse::kUniform) {
            context.value_kind = pto_analysis::CastValueKind::kScalar;
            context.logical_lanes = 1;
          } else {
            context.value_kind = pto_analysis::CastValueKind::kVectorVmi;
            context.logical_lanes = parent_->mapping_.lanes;
          }
          std::string why;
          auto plan = pto_analysis::PlanValueCast(op, context, &why);
          if (!plan.has_value()) {
            std::ostringstream oss;
            oss << "Cast " << op->value.dtype() << " -> " << op->dtype
                << " has no supported conversion";
            if (!why.empty()) {
              oss << ": " << why;
            }
            parent_->parent_->Fail("dtype", oss.str());
          }
          // Nested value casts each plan independently; annotations are not
          // computation values and are validated by the planner.
          VisitExpr(op->value);
        }
        void VisitExpr_(const BufferLoadNode *) final {
          // Index casts belong to address analysis; do not descend.
        }

      private:
        UnitVisitor *parent_;
        bool indexed_offset_;
      };
      CastCheck checker(this, indexed_offset);
      checker(expr);
    }

    void VerifyAccess(const Buffer &buffer, const Array<PrimExpr> &indices,
                      bool is_write, const Optional<PrimExpr> &predicate) {
      // Predicated accesses have no supported lowering; only
      // absent or provably-true predicates are accepted.
      if (predicate.defined()) {
        // The predicate is an expression position too — a
        // provably-true predicate (e.g. `True || ghost > 0`) could hide
        // undefined vars that Vectorize then silently drops with the
        // predicate. Run the same reference check as every other
        // position before the true-check.
        CheckDefinedVars(predicate.value());
        arith::Analyzer probe;
        PrimExpr simplified = probe.Simplify(predicate.value());
        if (!is_one(simplified)) {
          std::ostringstream oss;
          oss << "buffer `" << buffer->name
              << "` has a non-trivial access predicate; predicated "
                 "load/store is not yet supported";
          parent_->Fail("memory access", oss.str());
        }
      }
      try {
        auto access = pto_analysis::AnalyzeBufferAccess(
            buffer, indices, is_write, loop_var_, mapping_, &analyzer_,
            bind_env_);
        if (access.pattern != pto_analysis::AccessPattern::kContinuous &&
            access.pattern != pto_analysis::AccessPattern::kLaneUniform &&
            access.pattern != pto_analysis::AccessPattern::kIndexed) {
          std::ostringstream oss;
          oss << "buffer `" << buffer->name << "` access pattern "
              << pto_analysis::AccessPatternToString(access.pattern)
              << " is not supported";
          parent_->Fail("memory access", oss.str());
        }
        if (access.pattern == pto_analysis::AccessPattern::kIndexed ||
            (access.pattern == pto_analysis::AccessPattern::kLaneUniform &&
             is_write)) {
          pto_analysis::IndexedMemoryOwner owner =
              pto_analysis::GetIndexedMemoryOwner(
                  buffer, parent_->active_allocations_, &parent_->analyzer_,
                  "VerifyParallelToPTO");
          pto_analysis::PtoIndexedAccessPlan plan =
              pto_analysis::PlanPtoIndexedAccess(access, owner, loop_var_,
                                                 mapping_, &parent_->analyzer_);
          pto_analysis::CheckPtoIndexedAccessPlan(
              plan, owner, loop_var_, mapping_, &parent_->analyzer_);
          const PrimExpr &relative = plan.offsets;
          const int bits = buffer->dtype.bits();
          if (bits != 16 && bits != 32) {
            parent_->Fail("dtype", "indexed gather/scatter requires a 16-bit "
                                   "or 32-bit payload");
          }
          // Verify exactly what LowerIndexed converts, after any proved B16
          // rebasing and removal of an ElemOffset-only promotion. Metadata
          // such as strides/elem_offset can introduce operations and Var
          // references absent from the source index array.
          CheckDefinedVars(relative);
          const std::string offset_context = "buffer `" +
                                             std::string(buffer->name) +
                                             "` complete indexed offset";
          if (!pto_analysis::IsPtoIndexDType(relative.dtype())) {
            parent_->Fail("dtype", offset_context +
                                       " requires a scalar 16-bit or 32-bit "
                                       "integer expression");
          }
          if (!pto_analysis::IsSupportedAddressExpr(relative)) {
            parent_->Fail("expression", offset_context +
                                            " contains an unsupported call "
                                            "or a non-UB index load");
          }
          // ElemOffset metadata can contain reads that were absent from the
          // source indices. Apply the same scope, predicate, conditional-read
          // and nested-access rules before ConvertExpr can consume them.
          auto check_metadata_reads = [&](const PrimExpr &expr) {
            RegionVisitor metadata_accesses(parent_);
            metadata_accesses(expr);
            ConditionalIndexedAccessVisitor conditional_accesses(
                parent_, loop_var_, mapping_, external_defs_, bind_env_);
            conditional_accesses(expr);
            VisitExpr(expr);
          };
          // Source indices are visited by the ordinary unit walk. Rewalking
          // the complete rebased offset here duplicates every nested gather
          // subtree, so validate only the metadata introduced by ElemOffset.
          // This also covers metadata moved into the scalar base by rebasing.
          check_metadata_reads(buffer->elem_offset);
          bool uses_outer_coordinate = false;
          for (size_t axis = 0; axis < indices.size(); ++axis) {
            if (buffer->strides.empty()) {
              if (uses_outer_coordinate) {
                check_metadata_reads(buffer->shape[axis]);
              }
              uses_outer_coordinate |= !is_zero(indices[axis]);
            } else if (!is_zero(indices[axis])) {
              check_metadata_reads(buffer->strides[axis]);
            }
          }
          std::string offset_why;
          if (ClassifyComputeOperand(relative, /*indexed_offset=*/true,
                                     &offset_why) ==
              pto_analysis::LaneUse::kUnknown) {
            parent_->Fail("expression",
                          offset_context +
                              " cannot be classified: " + offset_why);
          }
          CheckValueCasts(relative, /*indexed_offset=*/true);
          CheckBinaryOpTypes(relative, /*indexed_offset=*/true);
        }
      } catch (const pto_analysis::PtoAnalysisError &err) {
        std::ostringstream oss;
        oss << "buffer `" << buffer->name << "`: " << err.what();
        parent_->Fail("memory access", oss.str());
      }
    }

    VerifyParallelToPTOImpl *parent_;
    const Var &loop_var_;
    /*! 2D only — the non-vectorized coordinate (uniform). */
    Var serial_var_;
    const pto_analysis::PtoInverseMapping &mapping_;
    arith::Analyzer analyzer_;
    Map<Var, PrimExpr> bind_env_;
    // Classification recorded at each Bind definition
    // (0 = uniform, 1 = varying) for this unit.
    Map<Var, Integer> classified_;
    // The lexical definition snapshot at the unit's position —
    // params, buffer data pointers, and every definition (outer loop
    // vars, thread_extent bindings, binds, allocations) that is in scope
    // where the unit appears.
    std::set<const VarNode *> external_defs_;
  };

  // Keep flat allocation definitions visible within the enclosing statement
  // sequence, then discard them when that lexical scope ends.
  void VisitStmt(const Stmt &stmt) final {
    if (stmt.as<AllocBufferNode>()) {
      StmtExprVisitor::VisitStmt(stmt);
      return;
    }
    const size_t allocation_mark = active_allocations_.size();
    StmtExprVisitor::VisitStmt(stmt);
    active_allocations_.resize(allocation_mark);
  }

  // ------------------------------------------------------------------
  // Scope tracking: every definition — outer serial loop vars,
  // thread_extent bindings, flat binds, buffer allocations/declarations
  // — enters the environment at its lexical position and leaves it when
  // the enclosing scope exits. A unit is verified against a snapshot of
  // this environment at the unit's position, so a sibling scope's
  // definitions can no longer leak in and legitimately-defined outer
  // scalars (blockIdx, outer binds) are no longer rejected.
  // ------------------------------------------------------------------
  void VisitStmt_(const ForNode *op) final {
    analyzer_.Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
    if (op->kind != ForKind::kParallel) {
      std::set<const VarNode *> saved = defs_;
      defs_.insert(op->loop_var.get());
      StmtExprVisitor::VisitStmt_(op);
      defs_ = std::move(saved);
      return;
    }
    // A parallel For outside a converting region is not this pass's
    // concern; its iteration var has no ordered single-assignment
    // definition to track here.
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tl::attr::kBufferVersion) {
      BufferVersionMap saved = buffer_versions_;
      for (const auto &[storage, count] : op->node.cast<BufferVersionMap>()) {
        buffer_versions_.Set(storage, count);
      }
      StmtExprVisitor::VisitStmt_(op);
      buffer_versions_ = std::move(saved);
      return;
    }
    if (op->attr_key == tirx::attr::thread_extent) {
      if (const auto *iv = op->node.as<IterVarNode>()) {
        // blockIdx/cthread bindings (e.g. `with T.Kernel(2) as bx`) are
        // legitimately-defined outer scalars for the attr body. The
        // no-threadIdx-inside-the-converting-region rule is
        // unaffected: it is enforced by the RegionVisitor inside the
        // region, not here.
        // Keep the launch range in the same analyzer used by CheckPtoFill;
        // otherwise a scalar value such as values[bx] is rejected even
        // though bx is provably in [0, thread_extent).
        With<arith::ConstraintContext> launch_bounds(
            &analyzer_, iv->var >= 0 && iv->var < op->value);
        std::set<const VarNode *> saved = defs_;
        defs_.insert(iv->var.get());
        StmtExprVisitor::VisitStmt_(op);
        defs_ = std::move(saved);
        return;
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BindNode *op) final {
    // Flat sequence semantics: the bound var is visible to all
    // subsequent statements of the enclosing scope; the scope owners
    // above restore the environment on exit.
    defs_.insert(op->var.get());
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AllocBufferNode *op) final {
    // Flat statement (no body): the allocation is visible to the rest of
    // the enclosing scope.
    defs_.insert(op->buffer->data.get());
    active_allocations_.push_back(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const DeclBufferNode *op) final {
    defs_.insert(op->buffer->data.get());
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    // Branch bodies are independent scopes.
    StmtExprVisitor::VisitExpr(op->condition);
    {
      std::set<const VarNode *> saved = defs_;
      VisitStmt(op->then_case);
      defs_ = std::move(saved);
    }
    if (op->else_case.defined()) {
      std::set<const VarNode *> saved = defs_;
      VisitStmt(op->else_case.value());
      defs_ = std::move(saved);
    }
  }

  void VisitStmt_(const WhileNode *op) final {
    StmtExprVisitor::VisitExpr(op->condition);
    std::set<const VarNode *> saved = defs_;
    StmtExprVisitor::VisitStmt(op->body);
    defs_ = std::move(saved);
  }

  void VisitStmt_(const SBlockNode *op) final {
    for (const Buffer &buffer : op->alloc_buffers) {
      active_allocations_.push_back(buffer);
    }
    if (op->name_hint != "SIMD_VF") {
      std::set<const VarNode *> saved = defs_;
      for (const auto &iv : op->iter_vars) {
        defs_.insert(iv->var.get());
      }
      for (const auto &buf : op->alloc_buffers) {
        defs_.insert(buf->data.get());
      }
      StmtExprVisitor::VisitStmt_(op);
      defs_ = std::move(saved);
      return;
    }
    HandleSimdVF(op);
  }

  // ------------------------------------------------------------------
  // SIMD_VF region entry (converting regions only)
  // ------------------------------------------------------------------
  void HandleSimdVF(const SBlockNode *op) {
    // Block-scoped definitions (allocations, iter vars) are in scope for
    // the whole region (supersedes the earlier simd_vf_allocs_ side
    // table).
    std::set<const VarNode *> saved = defs_;
    for (const auto &iv : op->iter_vars) {
      defs_.insert(iv->var.get());
    }
    for (const auto &buf : op->alloc_buffers) {
      defs_.insert(buf->data.get());
    }

    // Shared region scan: same implementation as Vectorize.
    const Stmt &region = op->body;
    pto_analysis::PtoRegionScan scan = pto_analysis::ScanPtoRegion(region);
    if (!scan.has_any_parallel && !scan.has_direct_reduce && !scan.has_fill) {
      // Pure hand-written VMI region (possibly with control flow): skipped
      // entirely; no lanes annotation required.
      defs_ = std::move(saved);
      return;
    }
    // Number the converting region for diagnostics. The lane count is
    // reset to unknown first: a failure before this region's own lanes
    // are read must not inherit the previous region's value.
    current_region_ = region_seq_++;
    current_lanes_ = 0;
    SetRegionContext();
    if (scan.has_unsupported_wrapper) {
      Fail("loop structure", scan.issue);
    }
    const std::vector<For> &units = scan.units;

    // Only now is the lanes annotation required.
    int64_t lanes = ReadLanes(op);
    current_lanes_ = lanes;
    SetRegionContext(); // refresh the region context with the lane count

    // Region-level checks across the whole converting region.
    RegionVisitor region_visitor(this);
    region_visitor(region);

    // Per-unit checks in order (multiple sequential units OK) against the
    // lexical definition snapshot at each unit's position.
    size_t next_unit = 0;
    WalkRegion(region, units, lanes, &next_unit);
    ICHECK(next_unit == units.size())
        << "[VerifyParallelToPTO] internal error: region walk and unit "
           "scan disagree on the unit count";
    defs_ = std::move(saved);
  }

  // ------------------------------------------------------------------
  // Per-unit checks (structure, ranges, layout)
  // ------------------------------------------------------------------
  void VerifyUnit(const For &unit, int64_t lanes, size_t unit_index) {
    // Full per-unit context for diagnostics.
    {
      std::ostringstream oss;
      oss << "region #" << current_region_ << ", unit #" << unit_index
          << " (loop var `" << unit->loop_var->name_hint << "`, extent "
          << unit->extent << ", lanes " << lanes << ")";
      unit_context_ = oss.str();
    }
    if (!is_zero(unit->min)) {
      std::ostringstream oss;
      oss << "loop min must be 0, got " << unit->min;
      Fail("loop range", oss.str());
    }
    // The lane layout assumes unit stride; a non-unit step
    // changes the set of iteration points and has no supported
    // lowering.
    if (unit->step.defined() &&
        !is_one(analyzer_.Simplify(unit->step.value()))) {
      std::ostringstream oss;
      oss << "T.Parallel step must be 1, got " << unit->step;
      Fail("loop range", oss.str());
    }
    const int64_t *extent = as_const_int(unit->extent);
    if (extent == nullptr || *extent <= 0) {
      std::ostringstream oss;
      oss << "loop extent must be a positive compile-time constant, got "
          << unit->extent;
      Fail("loop range", oss.str());
    }
    if (!unit->annotations.count(attr::kPtoParallelLoopLayout)) {
      Fail("layout", "pto_parallel_loop_layout is missing on the "
                     "outermost T.Parallel");
    }

    // Identify the full loop nest and check every member loop
    // *before* the inversion helper (which relies on constant extents) so
    // an illegal inner loop gets its loop-range diagnostic instead of a
    // raw assertion.
    const ForNode *inner_probe = nullptr;
    if (const auto *body_for = unit->body.as<ForNode>()) {
      if (body_for->kind == ForKind::kParallel) {
        inner_probe = body_for;
      }
    }
    if (inner_probe != nullptr) {
      if (!is_zero(inner_probe->min)) {
        std::ostringstream oss;
        oss << "inner loop min must be 0, got " << inner_probe->min;
        Fail("loop range", oss.str());
      }
      if (inner_probe->step.defined() &&
          !is_one(analyzer_.Simplify(inner_probe->step.value()))) {
        std::ostringstream oss;
        oss << "inner T.Parallel step must be 1, got " << inner_probe->step;
        Fail("loop range", oss.str());
      }
      if (as_const_int(inner_probe->extent) == nullptr) {
        std::ostringstream oss;
        oss << "inner loop extent must be a compile-time constant, got "
            << inner_probe->extent;
        Fail("loop range", oss.str());
      } else if (*as_const_int(inner_probe->extent) <= 0) {
        std::ostringstream oss;
        oss << "inner loop extent must be a positive compile-time constant, "
               "got "
            << inner_probe->extent;
        Fail("loop range", oss.str());
      }
      if (inner_probe->loop_var.dtype() != DataType::Int(32)) {
        std::ostringstream oss;
        oss << "inner loop variable must be int32 (the dtype "
               "LayoutInference produces), got "
            << inner_probe->loop_var.dtype();
        Fail("loop range", oss.str());
      }
    }

    pto_analysis::PtoInverseMapping mapping = [&] {
      try {
        return pto_analysis::InvertPtoLaneLayout(unit);
      } catch (const pto_analysis::PtoAnalysisError &err) {
        Fail("layout", err.what());
        return pto_analysis::PtoInverseMapping();
      }
    }();
    if (mapping.lanes != lanes) {
      std::ostringstream oss;
      oss << "layout lanes (" << mapping.lanes
          << ") disagree with tl.simdvf_lanes (" << lanes << ")";
      Fail("lanes", oss.str());
    }

    // A 2D unit is the outermost For of a two-layer adjacent
    // Parallel nest; the unit visitor runs on the innermost compute body
    // (the layout annotation only sits on the outer For). The
    // inner loop's range contract was already enforced by the
    // pre-inversion block above (min/step/extent/dtype), so no per-loop
    // check is repeated here.
    const ForNode *inner = nullptr;
    if (const auto *body_for = unit->body.as<ForNode>()) {
      if (body_for->kind == ForKind::kParallel) {
        inner = body_for;
      }
    }

    // The vectorized coordinate is the selected dimension of a 2D mapping
    // and the single loop var in 1D (the other 2D coordinate stays a
    // serial loop var and classifies as uniform).
    const Var vector_var =
        mapping.is_2d
            ? (mapping.select_inner ? mapping.inner_var : mapping.outer_var)
            : unit->loop_var;
    // The definition snapshot at the unit's lexical position.
    const Stmt &body_to_verify = inner != nullptr ? inner->body : unit->body;
    ConditionalIndexedAccessVisitor conditional_accesses(this, vector_var,
                                                         mapping, defs_);
    conditional_accesses(body_to_verify);
    UnitVisitor unit_visitor(this, vector_var, mapping, defs_);
    unit_visitor(body_to_verify);
  }

  // Region-level statements (between/around the Parallel units)
  // previously only *registered* definitions without scanning their
  // expressions, so an undefined var in a region-level Bind RHS (or
  // AttrStmt metadata, or serial-For bounds) leaked into the output
  // unchecked. Check that every Var is defined in the current lexical
  // environment; call sites run this *before* registering new
  // definitions so lexical order is preserved (a self reference is
  // undefined too).
  void CheckRegionExprDefined(const PrimExpr &expr, const char *what) {
    PostOrderVisit(expr, [&](const ObjectRef &node) {
      if (const auto *v = node.as<VarNode>()) {
        if (defs_.count(v) == 0) {
          std::ostringstream oss;
          oss << what << " references variable `" << v->name_hint
              << "` which is not defined in the lexical scope of this "
                 "SIMD_VF region";
          Fail("expression", oss.str());
        }
      }
    });
  }

  // Walk the converting region in statement order, mirroring the wrapper
  // kinds DiscoverUnits descends (For/SeqStmt/AttrStmt/SBlock/
  // SBlockRealize). Definitions before a unit (flat binds, serial loop
  // vars, allocations) are registered as encountered so each unit is
  // verified against the environment at its position. Parallel Fors are
  // the discovered units themselves; IfThenElse/While cannot contain
  // units (the scan rejected that), and unit bodies belong to the
  // UnitVisitor.
  void WalkRegion(const Stmt &stmt, const std::vector<For> &units,
                  int64_t lanes, size_t *next_unit) {
    // This walk does not use VisitStmt, so maintain the same allocation
    // scope here. A flat AllocBuffer remains visible to later siblings;
    // allocations inside every other statement expire on leaving it.
    struct AllocationScope {
      std::vector<Buffer> *allocations;
      size_t mark;
      bool restore;
      ~AllocationScope() {
        if (restore) {
          allocations->resize(mark);
        }
      }
    } allocation_scope{&active_allocations_, active_allocations_.size(),
                       stmt.as<AllocBufferNode>() == nullptr};
    if (const auto *for_node = stmt.as<ForNode>()) {
      if (for_node->kind == ForKind::kParallel &&
          !pto_analysis::UsesExplicitSIMD(for_node->body)) {
        ICHECK(*next_unit < units.size() && units[*next_unit].get() == for_node)
            << "[VerifyParallelToPTO] internal error: region walk and "
               "unit scan disagree on unit order";
        VerifyUnit(units[*next_unit], lanes, *next_unit);
        ++*next_unit;
        return;
      }
      // The serial loop bounds are expression positions too.
      // The step is one as well (the base statement visitors do
      // not descend into For::step).
      SetRegionContext();
      CheckRegionExprDefined(for_node->min, "serial loop min");
      CheckRegionExprDefined(for_node->extent, "serial loop extent");
      if (for_node->step.defined()) {
        CheckRegionExprDefined(for_node->step.value(), "serial loop step");
      }
      std::set<const VarNode *> saved = defs_;
      defs_.insert(for_node->loop_var.get());
      analyzer_.Bind(for_node->loop_var,
                     Range::FromMinExtent(for_node->min, for_node->extent));
      WalkRegion(for_node->body, units, lanes, next_unit);
      defs_ = std::move(saved);
      return;
    }
    if (const auto *assertion = stmt.as<AssertStmtNode>()) {
      // A region-level assertion is an expression position: its condition
      // must reference defined values. The node has no body (only the
      // condition, error kind and message parts), so nothing else is
      // walked; an assertion *inside* a unit stays rejected by the unit
      // statement visitor.
      SetRegionContext();
      CheckRegionExprDefined(assertion->condition,
                             "region-level assert condition");
      return;
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const auto &s : seq->seq) {
        WalkRegion(s, units, lanes, next_unit);
      }
      return;
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      // Region-level metadata (node/value) is scanned for
      // undefined references before descending.
      SetRegionContext();
      if (auto ne = attr->node.try_cast<ObjectRef>()) {
        if (const auto *pe = ne.value().as<PrimExprNode>()) {
          CheckRegionExprDefined(GetRef<PrimExpr>(pe),
                                 "region-level AttrStmt metadata");
        }
      }
      CheckRegionExprDefined(attr->value, "region-level AttrStmt metadata");
      // The body is a nested statement scope.
      {
        std::set<const VarNode *> saved = defs_;
        WalkRegion(attr->body, units, lanes, next_unit);
        defs_ = std::move(saved);
      }
      return;
    }
    if (const auto *block = stmt.as<SBlockNode>()) {
      std::set<const VarNode *> saved = defs_;
      for (const auto &iv : block->iter_vars) {
        defs_.insert(iv->var.get());
      }
      for (const auto &buf : block->alloc_buffers) {
        defs_.insert(buf->data.get());
        active_allocations_.push_back(buf);
      }
      WalkRegion(block->body, units, lanes, next_unit);
      defs_ = std::move(saved);
      return;
    }
    if (const auto *realize = stmt.as<SBlockRealizeNode>()) {
      std::set<const VarNode *> saved = defs_;
      for (const auto &iv : realize->block->iter_vars) {
        defs_.insert(iv->var.get());
      }
      for (const auto &buf : realize->block->alloc_buffers) {
        defs_.insert(buf->data.get());
        active_allocations_.push_back(buf);
      }
      WalkRegion(realize->block->body, units, lanes, next_unit);
      defs_ = std::move(saved);
      return;
    }
    if (const auto *bind = stmt.as<BindNode>()) {
      // The region-level Bind RHS is checked against the
      // pre-registration environment (lexical order), then the
      // definition itself is registered.
      SetRegionContext();
      CheckRegionExprDefined(bind->value, "region-level bind");
      PrimExpr scalar_source = bind->value;
      bool converts_float = false;
      while (const auto *conversion = scalar_source.as<CastNode>()) {
        DataType from = conversion->value.dtype();
        DataType to = conversion->dtype;
        if (from.is_scalar() && to.is_scalar() && from != to &&
            (from.is_float() || from.is_bfloat16()) &&
            (to.is_float() || to.is_bfloat16())) {
          converts_float = true;
        }
        scalar_source = conversion->value;
      }
      if (converts_float && scalar_source.as<BufferLoadNode>()) {
        Fail("dtype",
             "region-level floating-point casts of UB reads have no scalar "
             "lowering inside SIMD_VF; put the casts directly in T.fill or "
             "use a lane-varying T.Parallel value");
      }
      defs_.insert(bind->var.get());
      return;
    }
    if (const auto *alloc = stmt.as<AllocBufferNode>()) {
      defs_.insert(alloc->buffer->data.get());
      active_allocations_.push_back(alloc->buffer);
      return;
    }
    if (const auto *decl = stmt.as<DeclBufferNode>()) {
      defs_.insert(decl->buffer->data.get());
      return;
    }
    // Statements that contribute no definitions are still
    // expression positions. A store, an Evaluate, a branch or a while
    // outside the Parallel units previously ended the walk silently, so
    // an undefined variable in any of their expression fields passed
    // Verify and stayed in the output.
    if (const auto *store = stmt.as<BufferStoreNode>()) {
      SetRegionContext();
      CheckRegionExprDefined(store->value, "region-level store value");
      for (const auto &idx : store->indices) {
        CheckRegionExprDefined(idx, "region-level store index");
      }
      if (store->predicate.defined()) {
        CheckRegionExprDefined(store->predicate.value(),
                               "region-level store predicate");
      }
      return;
    }
    if (const auto *eval = stmt.as<EvaluateNode>()) {
      SetRegionContext();
      CheckRegionExprDefined(eval->value, "region-level evaluate value");
      if (const auto *call_node = eval->value.as<CallNode>()) {
        Call call = GetRef<Call>(call_node);
        if (pto_analysis::IsPtoDirectReduce(call)) {
          try {
            pto_analysis::CheckPtoDirectReduce(call, &analyzer_,
                                               buffer_versions_);
          } catch (const pto_analysis::PtoAnalysisError &error) {
            Fail("reduce", error.what());
          }
        } else if (pto_analysis::IsPtoFill(call)) {
          try {
            pto_analysis::CheckPtoFill(call, lanes, &analyzer_,
                                       buffer_versions_);
          } catch (const pto_analysis::PtoAnalysisError &error) {
            Fail("fill/clear", error.what());
          }
        }
      }
      return;
    }
    if (const auto *ite = stmt.as<IfThenElseNode>()) {
      // A Parallel under control flow is rejected by ScanPtoRegion, so
      // the branches carry plain statements only; branch-local
      // definitions must not leak to later siblings.
      SetRegionContext();
      CheckRegionExprDefined(ite->condition, "region-level branch condition");
      {
        std::set<const VarNode *> saved = defs_;
        WalkRegion(ite->then_case, units, lanes, next_unit);
        defs_ = std::move(saved);
      }
      if (ite->else_case.defined()) {
        std::set<const VarNode *> saved = defs_;
        WalkRegion(ite->else_case.value(), units, lanes, next_unit);
        defs_ = std::move(saved);
      }
      return;
    }
    if (const auto *while_node = stmt.as<WhileNode>()) {
      SetRegionContext();
      CheckRegionExprDefined(while_node->condition,
                             "region-level while condition");
      std::set<const VarNode *> saved = defs_;
      WalkRegion(while_node->body, units, lanes, next_unit);
      defs_ = std::move(saved);
      return;
    }
  }

  arith::Analyzer analyzer_;
  // The lexical definition environment. Supersedes the earlier
  // function-wide external-def union, which missed scoped definitions
  // (blockIdx, outer/VF-internal binds) and leaked sibling-scope loop
  // vars into every unit.
  std::set<const VarNode *> defs_;
  // Data Vars of AllocBuffers annotated tirx.volatile (collected
  // function-wide by CollectVolatileAllocs; identity-matched at every
  // access inside a converting region).
  std::set<const VarNode *> volatile_allocs_;
  // Position of the converting region/unit being checked, for
  // diagnostics.
  int region_seq_ = 0;
  int current_region_ = -1;
  std::string unit_context_;
  // Lane count of the converting region under check (0 when unknown); part of
  // the region/unit diagnostic context.
  int64_t current_lanes_ = 0;
  std::vector<Buffer> active_allocations_;
  BufferVersionMap buffer_versions_;
};

} // namespace

namespace transform {

tvm::transform::Pass VerifyParallelToPTO() {
  auto pass_func = [](PrimFunc f, const IRModule &m,
                      const tvm::transform::PassContext &ctx) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsPTO(target.value())) {
      return f;
    }
    VerifyParallelToPTOImpl verifier;
    verifier.Verify(f);
    return f;
  };
  return tvm::tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                                  "tl.VerifyParallelToPTO", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.VerifyParallelToPTO",
                        VerifyParallelToPTO);
}

} // namespace transform

} // namespace tl

} // namespace tvm
