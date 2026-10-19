// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file vectorize_parallel_to_pto.cc
 * \brief VectorizeParallelToPTO: rewrite verified T.Parallel loops inside
 * PTO SIMD_VF blocks into logical VMI code.
 *
 * Consumes the pto_parallel_loop_layout fragment written by
 * LayoutInference and verified by VerifyParallelToPTO. Per parallel unit:
 *   1. invert the lane fragment through the shared helper (the inverse
 *      layout's own Forward result is the single source of truth);
 *   2. build the chunk loop for q in [0, E/L), mapping the loop variable
 *      to q * L + lane;
 *   3. classify every access via the shared MemoryAccess analysis
 *      (continuous -> vload/vstore via tl.access_ptr(start, L, rw);
 *      lane-uniform -> scalar BufferLoad with lane bound to 0);
 *   4. convert value expressions (value roles tracked by result dtype);
 *      every VMI with a return value is bound through EmitBind before use;
 *   5. delete the consumed layout annotation, keep the SIMD_VF block
 *      annotations, and leave no unconverted Parallel behind.
 *
 * Supported shapes: 1D parallel units and two-layer adjacent Parallel nests
 * (2D — the nest converts as one unit with the layout
 * annotation on its outermost For); sequential units allowed; static
 * divisible extents; full masks. A third parallel layer or any other
 * unsupported structure is rejected by Verify.
 *
 * Non-divisible extents emit a tail chunk whose vload reads the full L
 * lanes from its chunk base: the ISA intentionally keeps load instructions
 * unmasked for performance, so the per-chunk tail mask constrains
 * computation and stores only, never the load. Reading the full L lanes
 * past the logical tail is still safe on device — the top 8K of UB is
 * reserved as stack space, so the extra read-only lanes cannot fault; they
 * carry undefined values that the compute and store masks keep out of the
 * valid outputs. Compact outputs of exactly E elements remain safe because
 * the masked store never writes past E.
 */

#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "parallel_to_pto_utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;
namespace pto_analysis = ::tvm::tl::pto;

namespace {

DataType VectorDType(DataType elem, int64_t lanes) {
  return elem.with_lanes(static_cast<int>(lanes));
}

const Op &VmiOp(const std::string &tail) { return Op::Get("tl.vmi." + tail); }

// This version keeps local DCE in Vectorize so
// its directly emitted TIR/PTODSL is cleaner and easier to inspect.
// Cleanup may later move to PTOAS canonicalization/DCE to reduce
// TileLang-specific code and maintenance; clean-output checks would then
// target the optimized PTOAS IR. The emitted IR must be valid and preserve
// program semantics even before DCE.
//
// Dead vector-chain elimination on one converted chunk body. The
// conversion can leave Bind statements whose Var is never referenced:
// source binds consumed only through the address alias environment
// (bind_env_) never use their converted Var, and an unused source bind
// strands its whole value chain (vci/vbrc/vadd/vload).
//
// The traversal is a scope-aware *reverse liveness* pass — a
// dead Bind must not survive just because it was wrapped in an AttrStmt
// or appears as the only statement of a body. The rules are:
//   1. A SeqStmt is processed back to front; `live` always holds the
//      vars everything *after* the current statement needs.
//   2. A Bind's LHS var is a definition, never a use. When the result is
//      not live and the RHS is discardable, the Bind is deleted and its
//      RHS inputs do NOT become live (so the dead chain collapses in one
//      pass); otherwise it is kept and its RHS inputs join `live`.
//   3. An AttrStmt body is cleaned in its own nested scope with a fresh
//      liveness set; only the vars actually referenced inside propagate
//      outward. Attribute node/value references are always uses, and the
//      attribute structure and evaluation position are preserved.
//   4. A bare Bind follows the same rules; when everything dies the
//      remains are a legal no-op (Evaluate(0)).
// Nothing is merged, reordered or otherwise rescheduled;
// referenced loads stay exactly where they were (vecadd keeps its three
// vloads), and Bind-free (frontend-inlined) bodies are untouched.
class DeadBindElim {
public:
  Stmt operator()(const Stmt &stmt) {
    std::set<const VarNode *> live;
    Stmt out = Run(stmt, &live);
    return out.defined() ? out : NoOp();
  }

private:
  static Stmt NoOp() { return Evaluate(make_zero(DataType::Int(32))); }

  static void CollectExprVars(const PrimExpr &expr,
                              std::set<const VarNode *> *live) {
    PostOrderVisit(expr, [&](const ObjectRef &n) {
      if (const auto *v = n.as<VarNode>()) {
        live->insert(v);
      }
    });
  }

  static void CollectStmtVars(const Stmt &stmt,
                              std::set<const VarNode *> *live) {
    PostOrderVisit(stmt, [&](const ObjectRef &n) {
      if (const auto *v = n.as<VarNode>()) {
        live->insert(v);
      }
    });
  }

  static bool DiscardableRhs(const PrimExpr &rhs) {
    // Positive "allowed to discard" set: a Call may disappear
    // only when it is one of the tl.vmi.* ops this pass actually emits as
    // a Bind RHS and that is pure/read-only by construction — vload
    // (read), vbrc, vci, vadd, vmul and vcvt (pure), enumerated from the
    // ConvertUnit/VmiOp call sites — or the pure tl.access_ptr address
    // descriptor nested inside a vload operand (already whitelisted as a
    // pure index intrinsic by IsSupportedAddressExpr). vstore never
    // appears as a Bind RHS; any other Call — call_extern, other tl.* or
    // an unknown op — keeps its Bind. Plain UB BufferLoads are
    // discardable: they are ordinary reads with no write effect (the
    // scalar "sload" binds). That "ordinary" premise is enforced by
    // VerifyParallelToPTO, not assumed here: accesses to
    // tirx.volatile-annotated allocations and volatile_scope attributes
    // are rejected there, and opaque calls are already excluded
    // from addresses.
    static const std::set<std::string> kDiscardable = {
        "tl.vmi.vload",
        "tl.vmi.vbrc",
        "tl.vmi.vci",
        "tl.vmi.vadd",
        "tl.vmi.vsub",
        "tl.vmi.vmul",
        "tl.vmi.vdiv",
        "tl.vmi.vabs",
        "tl.vmi.vexp",
        "tl.vmi.vln",
        "tl.vmi.vsqrt",
        "tl.vmi.vcvt",
        "tl.vmi.vinterpret_cast",
        // M2 control-flow ops (vcmp/vcmps produce masks, vsel/mask_* combine
        // them); an unused result is discardable like the arithmetic ops.
        "tl.vmi.vcmp",
        "tl.vmi.vcmps",
        "tl.vmi.vsel",
        "tl.vmi.mask_and",
        "tl.vmi.mask_or",
        "tl.vmi.mask_not",
        "tl.access_ptr",
    };
    bool discardable = true;
    PostOrderVisit(rhs, [&](const ObjectRef &n) {
      if (const auto *call = n.as<CallNode>()) {
        const auto *op = call->op.as<OpNode>();
        const std::string name = op != nullptr ? op->name : "";
        if (kDiscardable.count(name) == 0) {
          discardable = false;
        }
      }
    });
    return discardable;
  }

  // Returns the kept statement, or an *undefined* Stmt when the statement
  // was deleted (only a Bind can be deleted). `live` is threaded from
  // later statements back to earlier ones.
  Stmt Run(const Stmt &stmt, std::set<const VarNode *> *live) {
    if (const auto *bind = stmt.as<BindNode>()) {
      if (live->count(bind->var.get()) == 0 && DiscardableRhs(bind->value)) {
        // Dead and safe to drop: the RHS inputs stay out of `live`, so a
        // chain feeding only this Bind collapses in the same pass.
        return Stmt();
      }
      // Kept: classic liveness — subtract the definition, add the uses.
      live->erase(bind->var.get());
      CollectExprVars(bind->value, live);
      return stmt;
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      Array<Stmt> kept;
      for (auto it = seq->seq.rbegin(); it != seq->seq.rend(); ++it) {
        Stmt sub = Run(*it, live);
        if (sub.defined()) {
          kept.push_back(sub);
        }
      }
      if (kept.empty()) {
        return NoOp();
      }
      // std::reverse does not accept the ffi Array iterators on this
      // toolchain; re-append in reverse instead.
      Array<Stmt> forward;
      for (auto it = kept.rbegin(); it != kept.rend(); ++it) {
        forward.push_back(*it);
      }
      return SeqStmt::Flatten(forward);
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      // Nested scope: clean the body with a fresh liveness set, then
      // propagate only the references that survive to its entry (external
      // uses; vars defined and used only inside never escape because the
      // converted Vars are unique objects). Attribute node/value
      // references are always uses and keep the structure.
      std::set<const VarNode *> body_live;
      Stmt body = Run(attr->body, &body_live);
      if (!body.defined()) {
        body = NoOp();
      }
      live->insert(body_live.begin(), body_live.end());
      if (auto ne = attr->node.try_cast<ObjectRef>()) {
        if (const auto *pe = ne.value().as<PrimExprNode>()) {
          CollectExprVars(GetRef<PrimExpr>(pe), live);
        }
      }
      CollectExprVars(attr->value, live);
      if (body.same_as(attr->body)) {
        return stmt;
      }
      return AttrStmt(attr->node, attr->attr_key, attr->value, body);
    }
    // Any other statement is always kept; all its Var occurrences are
    // uses.
    CollectStmtVars(stmt, live);
    return stmt;
  }
};

class VectorizeParallelToPTOImpl : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc f) {
    VectorizeParallelToPTOImpl mutator;
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = mutator.VisitStmt(f->body);
    return GetRef<PrimFunc>(fptr);
  }

private:
  // ------------------------------------------------------------------
  // Unified unit discovery (shared rules with Verify).
  // A SIMD_VF region contains zero or more *sequential* parallel units.
  // Each unit is the outermost parallel For of a 1D loop or of a
  // two-layer adjacent Parallel nest (2D); a third parallel layer (or a
  // Parallel under an unsupported wrapper) is rejected by Verify and
  // defensively rejected here too.
  // ------------------------------------------------------------------
  // ------------------------------------------------------------------
  // SIMD_VF block entry
  // ------------------------------------------------------------------
  Stmt VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint != "SIMD_VF") {
      return StmtExprMutator::VisitStmt_(op);
    }
    // Shared region scan: the same implementation Verify uses, so
    // skip/reject/convert decisions cannot drift apart.
    const Stmt &region = op->body;
    pto_analysis::PtoRegionScan scan = pto_analysis::ScanPtoRegion(region);
    if (!scan.has_any_parallel) {
      // Pure hand-written VMI region (possibly with control flow): skipped
      // by both passes; no lanes annotation needed.
      return StmtExprMutator::VisitStmt_(op);
    }
    ICHECK(!scan.has_unsupported_wrapper)
        << "[VectorizeParallelToPTO] internal error: unsupported wrapper "
           "structure passed Verify: "
        << scan.issue;
    ICHECK(!scan.units.empty())
        << "[VectorizeParallelToPTO] internal error: parallel units lost "
           "after scan";

    Stmt new_body;
    {
      // One inverse per unit: the same mapping feeds the lanes read, the
      // shared-mask decision and the unit conversion below, so a region
      // with n units inverts n times instead of 2n+1. The inversion is
      // the expensive part (layout reshape + analyzer proofs); the
      // mappings are small value structs.
      std::vector<pto_analysis::PtoInverseMapping> mappings;
      mappings.reserve(scan.units.size());
      for (const For &unit : scan.units) {
        mappings.push_back(pto_analysis::InvertPtoLaneLayout(unit));
      }
      // Lanes come from the first unit's layout; all units in one VF
      // share the same L (Verify enforces the annotation consistency).
      int64_t lanes = mappings[0].lanes;
      // A full mask binding per VF, shared by the divisible units. A
      // region whose units all have non-divisible extents never uses it
      // (every unit builds its own per-chunk dynamic mask), so the
      // binding is only emitted when at least one divisible unit needs
      // it.
      bool any_divisible = false;
      for (const pto_analysis::PtoInverseMapping &mapping : mappings) {
        if (mapping.tail_lanes == 0) {
          any_divisible = true;
          break;
        }
      }
      Stmt new_region = region;
      if (any_divisible) {
        shared_mask_ = Var("mask", DataType::Bool(static_cast<int>(lanes)));
        PrimExpr mask_value = Call(
            DataType::Bool(static_cast<int>(lanes)), VmiOp("create_mask"),
            {IntImm(DataType::Int(32), static_cast<int>(lanes))},
            {{"size", IntImm(DataType::Int(32), static_cast<int>(lanes))}});
        // Define the shared mask inside the simdvf scope attr when the
        // region is wrapped by one: VPTO treats vecscope-region data as
        // region-local and rejects pset values crossing the boundary.
        const auto *scope_attr = new_region.as<AttrStmtNode>();
        if (scope_attr != nullptr &&
            scope_attr->attr_key == "tl.simdvf_scope") {
          new_region = AttrStmt(scope_attr->node, scope_attr->attr_key,
                                scope_attr->value,
                                SeqStmt::Flatten(Bind(shared_mask_, mask_value),
                                                 scope_attr->body));
        } else {
          new_region =
              SeqStmt::Flatten(Bind(shared_mask_, mask_value), new_region);
        }
      }

      // Convert each sequential unit in order; unit state (bind
      // maps, counters) is reset per unit; the shared full mask, when
      // emitted, is visible to every unit.
      for (size_t i = 0; i < scan.units.size(); ++i) {
        Stmt converted = ConvertUnit(scan.units[i], mappings[i], region);
        new_region = ReplaceStmt(new_region, scan.units[i], converted);
      }
      new_body = new_region;
    }

    // new_body is (mask bind + region with all units converted); the
    // scope AttrStmt wrapper, if any, is inside it already.
    return SBlock(op->iter_vars, op->reads, op->writes, op->name_hint, new_body,
                  op->init, op->alloc_buffers, op->match_buffers,
                  op->annotations);
  }

  // Replace one statement (the unit's For) with its conversion, recursing
  // through the same wrapper kinds DiscoverUnits understands.
  Stmt ReplaceStmt(const Stmt &body, const For &target,
                   const Stmt &replacement) {
    if (const auto *for_node = body.as<ForNode>()) {
      if (for_node == target.get()) {
        return replacement;
      }
      Stmt nb = ReplaceStmt(for_node->body, target, replacement);
      if (nb.same_as(for_node->body)) {
        return body;
      }
      For nf = GetRef<For>(for_node);
      nf.CopyOnWrite()->body = nb;
      return nf;
    }
    if (const auto *seq = body.as<SeqStmtNode>()) {
      Array<Stmt> new_seq;
      bool changed = false;
      for (const auto &s : seq->seq) {
        Stmt ns = ReplaceStmt(s, target, replacement);
        changed |= !ns.same_as(s);
        new_seq.push_back(ns);
      }
      return changed ? SeqStmt::Flatten(new_seq) : body;
    }
    if (const auto *attr = body.as<AttrStmtNode>()) {
      Stmt nb = ReplaceStmt(attr->body, target, replacement);
      if (nb.same_as(attr->body)) {
        return body;
      }
      return AttrStmt(attr->node, attr->attr_key, attr->value, nb);
    }
    if (const auto *block = body.as<SBlockNode>()) {
      Stmt nb = ReplaceStmt(block->body, target, replacement);
      if (nb.same_as(block->body)) {
        return body;
      }
      return SBlock(block->iter_vars, block->reads, block->writes,
                    block->name_hint, nb, block->init, block->alloc_buffers,
                    block->match_buffers, block->annotations);
    }
    if (const auto *realize = body.as<SBlockRealizeNode>()) {
      // nb is the replaced *body* of the block (a SeqStmt/For/etc.), not an
      // SBlock: rebuild the block with the new body first.
      SBlock block = realize->block;
      Stmt nb = ReplaceStmt(block->body, target, replacement);
      if (nb.same_as(block->body)) {
        return body;
      }
      SBlock new_block =
          SBlock(block->iter_vars, block->reads, block->writes,
                 block->name_hint, nb, block->init, block->alloc_buffers,
                 block->match_buffers, block->annotations);
      return SBlockRealize(realize->iter_values, realize->predicate, new_block);
    }
    return body;
  }

  // ------------------------------------------------------------------
  // Loop shells (constant-bound T.serial inside T.Parallel, admitted by
  // Verify). AnalyzeLoopShell runs per unit before conversion. The shell
  // itself is always preserved as a scalar loop; when the canonical
  // carry pattern matches, the carried value is promoted to a local.var
  // vreg buffer (the UB round-trip is not order-safe on the current
  // PTOAS: no automatic mem_bar between the carried vstore and the next
  // iteration's vload). A shell whose body both reads and writes a
  // buffer through UB without matching the pattern would silently read
  // stale values, so it is rejected instead.
  // ------------------------------------------------------------------
  static void FlattenStmts(const Stmt &stmt, std::vector<Stmt> *out) {
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const auto &s : seq->seq) {
        FlattenStmts(s, out);
      }
      return;
    }
    out->push_back(stmt);
  }

  class ShellAccessCollector : public StmtExprVisitor {
  public:
    struct Access {
      const BufferNode *buf;
      Array<PrimExpr> indices;
      bool is_store;
    };
    std::vector<Access> accesses;
    std::vector<const VarNode *> loop_vars;
    void VisitStmt_(const BufferStoreNode *op) final {
      accesses.push_back({op->buffer.get(), op->indices, true});
      StmtExprVisitor::VisitStmt_(op);
    }
    void VisitExpr_(const BufferLoadNode *op) final {
      accesses.push_back({op->buffer.get(), op->indices, false});
      StmtExprVisitor::VisitExpr_(op);
    }
    void VisitStmt_(const ForNode *op) final {
      loop_vars.push_back(op->loop_var.get());
      StmtExprVisitor::VisitStmt_(op);
    }
  };

  class BufferAccessCounter : public StmtExprVisitor {
  public:
    explicit BufferAccessCounter(const BufferNode *buf) : buf_(buf) {}
    void VisitStmt_(const BufferStoreNode *op) final {
      if (op->buffer.get() == buf_) {
        ++count_;
      }
      StmtExprVisitor::VisitStmt_(op);
    }
    void VisitExpr_(const BufferLoadNode *op) final {
      if (op->buffer.get() == buf_) {
        ++count_;
      }
      StmtExprVisitor::VisitExpr_(op);
    }
    int count_ = 0;

  private:
    const BufferNode *buf_;
  };

  [[noreturn]] void ShellFail(const std::string &message) {
    LOG(FATAL) << "[VectorizeParallelToPTO] loop shell: " << message;
  }

  bool IsCarryAccess(const Buffer &buf, const Array<PrimExpr> &indices) const {
    if (!carry_src_.defined() || !buf->data.same_as(carry_src_->data)) {
      return false;
    }
    if (indices.size() != carry_indices_.size()) {
      return false;
    }
    for (size_t i = 0; i < indices.size(); ++i) {
      if (!ffi::StructuralEqual()(indices[i], carry_indices_[i])) {
        return false;
      }
    }
    return true;
  }

  void AnalyzeLoopShell(const Stmt &unit_body, const Stmt &region_body) {
    carry_seed_ = nullptr;
    carry_loop_ = nullptr;
    carry_src_ = tirx::Buffer();
    carry_vreg_ = tirx::Buffer();
    carry_indices_ = Array<PrimExpr>();
    carry_active_ = false;

    std::vector<Stmt> stmts;
    FlattenStmts(unit_body, &stmts);
    const ForNode *shell = nullptr;
    size_t shell_pos = 0;
    for (size_t i = 0; i < stmts.size(); ++i) {
      if (const auto *f = stmts[i].as<ForNode>()) {
        if (f->kind == ForKind::kSerial) {
          if (shell != nullptr) {
            ShellFail("only one T.serial loop per T.Parallel unit is "
                      "supported at this stage");
          }
          shell = f;
          shell_pos = i;
        }
      }
    }
    if (shell == nullptr) {
      return; // no shell in this unit
    }

    ShellAccessCollector collector;
    collector(shell->body);
    // Buffers the shell both reads and writes through memory — the
    // cross-iteration dependencies.
    std::unordered_map<const BufferNode *, std::pair<int, int>> counts;
    for (const auto &acc : collector.accesses) {
      auto &c = counts[acc.buf];
      if (acc.is_store) {
        ++c.second;
      } else {
        ++c.first;
      }
    }
    const BufferNode *carried = nullptr;
    for (const auto &kv : counts) {
      if (kv.second.first > 0 && kv.second.second > 0) {
        if (carried != nullptr) {
          ShellFail("the T.serial loop reads and writes more than one "
                    "buffer; only a single carried buffer is supported at "
                    "this stage");
        }
        carried = kv.first;
      }
    }
    if (carried == nullptr) {
      // Shell without a memory-carried value: plain preservation has no
      // cross-iteration write->read through UB to order.
      return;
    }

    // Every access of the carried buffer inside the shell must use one
    // index set (structurally equal, independent of every loop
    // variable), with exactly one store.
    std::vector<const ShellAccessCollector::Access *> carried_accesses;
    for (const auto &acc : collector.accesses) {
      if (acc.buf == carried) {
        carried_accesses.push_back(&acc);
      }
    }
    int carried_stores = 0;
    for (const auto *acc : carried_accesses) {
      if (acc->is_store) {
        ++carried_stores;
      }
    }
    if (carried_stores != 1) {
      ShellFail("the carried buffer must have exactly one store inside the "
                "T.serial loop");
    }
    const Array<PrimExpr> &ref = carried_accesses[0]->indices;
    for (const auto *acc : carried_accesses) {
      if (acc->indices.size() != ref.size()) {
        ShellFail("all carried accesses must use the same index");
      }
      for (size_t i = 0; i < ref.size(); ++i) {
        if (!ffi::StructuralEqual()(acc->indices[i], ref[i])) {
          ShellFail("all carried accesses must use the same index");
        }
      }
    }
    std::vector<const VarNode *> loop_vars = collector.loop_vars;
    loop_vars.push_back(shell->loop_var.get());
    for (const auto &idx : ref) {
      for (const VarNode *lv : loop_vars) {
        if (tirx::UsesVar(idx, [lv](const VarNode *v) { return v == lv; })) {
          ShellFail("the carried index must not depend on the loop "
                    "variable");
        }
      }
    }

    // The seed: a store to the carried buffer, immediately before the
    // shell, whose value does not read the carried buffer.
    if (shell_pos == 0) {
      ShellFail("the loop carry needs a seed store to the carried buffer "
                "immediately before the T.serial loop");
    }
    const auto *seed = stmts[shell_pos - 1].as<BufferStoreNode>();
    if (seed == nullptr || seed->buffer.get() != carried ||
        seed->indices.size() != ref.size()) {
      ShellFail("the loop carry needs a seed store to the carried buffer "
                "immediately before the T.serial loop");
    }
    for (size_t i = 0; i < ref.size(); ++i) {
      if (!ffi::StructuralEqual()(seed->indices[i], ref[i])) {
        ShellFail("the seed store must use the same index as the carried "
                  "accesses");
      }
    }
    class SeedReadsCarried : public StmtExprVisitor {
    public:
      explicit SeedReadsCarried(const BufferNode *buf) : buf_(buf) {}
      void VisitExpr_(const BufferLoadNode *op) final {
        if (op->buffer.get() == buf_) {
          reads_ = true;
        }
        StmtExprVisitor::VisitExpr_(op);
      }
      bool reads_ = false;

    private:
      const BufferNode *buf_;
    };
    SeedReadsCarried seed_check(carried);
    seed_check(seed->value);
    if (seed_check.reads_) {
      ShellFail("the seed value must not read the carried buffer");
    }

    // Isolation: the carried buffer must not be accessed anywhere else
    // in the region (every other access would observe the eliminated
    // round-trip). The pattern's own accesses are the seed plus the
    // in-shell ones; the post-shell writeback is emitted, not counted.
    BufferAccessCounter region_count(carried);
    region_count(region_body);
    if (region_count.count_ !=
        static_cast<int>(carried_accesses.size()) + 1) {
      ShellFail("the carried buffer must only be accessed by the seed "
                "store and the T.serial loop in this SIMD_VF region");
    }

    carry_seed_ = seed;
    carry_loop_ = shell;
    carry_src_ = GetRef<Buffer>(carried);
    carry_indices_ = ref;
    carry_vreg_ = tirx::decl_buffer(
        {IntImm(DataType::Int(32), 1)},
        VectorDType(carried->dtype, lanes_),
        std::string(carried->name) + "_carry", "local.var");
  }

  Stmt WrapCarryAlloc(Stmt s) {
    if (carry_vreg_.defined()) {
      return SeqStmt::Flatten(AllocBuffer(carry_vreg_), std::move(s));
    }
    return s;
  }

  // The shell's own stores were eliminated by promotion; publish the
  // final carry value back to the source buffer once, after the shell.
  Stmt EmitCarryWriteback() {
    auto access = pto_analysis::AnalyzeBufferAccess(
        carry_src_, carry_indices_, /*is_write=*/true, loop_var_,
        CurrentMapping(), &analyzer_, bind_env_);
    ICHECK(access.pattern == pto_analysis::AccessPattern::kContinuous)
        << "[VectorizeParallelToPTO] internal error: the carry writeback "
           "address is not continuous";
    Array<PrimExpr> start_idx;
    for (const auto &idx : access.start_indices) {
      start_idx.push_back(
          analyzer_.Simplify(BindChunkToQ(SubstituteLaneZero(idx))));
    }
    PrimExpr ptr =
        Call(DataType::Handle(), tl::access_ptr(),
             {BufferLoad(carry_src_, start_idx),
              IntImm(DataType::Int(32), static_cast<int>(lanes_)),
              IntImm(DataType::Int(32), 2)});
    return Evaluate(Call(
        DataType::Void(), VmiOp("vstore"),
        {BufferLoad(carry_vreg_, {make_zero(DataType::Int(32))}), ptr,
         IntImm(DataType::Int(32), 0), mask_var_}));
  }

  // ------------------------------------------------------------------
  // Per-unit conversion
  // ------------------------------------------------------------------
  Stmt ConvertUnit(const For &loop,
                   const pto_analysis::PtoInverseMapping &mapping,
                   const Stmt &region_body) {
    // The inverse mapping comes from the region entry: it is the shared
    // source of truth for lanes, the mask plan and the body conversion
    // (see InvertPtoLaneLayout).
    // Reset per-unit conversion state.
    bind_env_.clear();
    value_map_.clear();
    tmp_counter_ = 0;
    // Keep the complete mapping (2D fields included) for the
    // per-access analysis calls in ConvertBody.
    unit_mapping_ = mapping;

    // For a 2D unit the vectorized coordinate is the selected
    // dimension; the other one stays a serial loop var. The converted
    // body is the innermost compute body.
    const ForNode *inner = nullptr;
    if (const auto *body_for = loop->body.as<ForNode>()) {
      if (body_for->kind == ForKind::kParallel) {
        inner = body_for;
      }
    }
    const Var vector_var =
        mapping.is_2d
            ? (mapping.select_inner ? mapping.inner_var : mapping.outer_var)
            : loop->loop_var;
    loop_var_ = vector_var;
    lanes_ = mapping.lanes;
    // The chunk loop covers the padded chunk count Q = P / L
    // (mapping.chunk_count), which includes the tail chunk of a
    // non-divisible extent.
    nchunks_ = mapping.chunk_count;
    chunk_var_ = Var("q", vector_var.dtype());
    lane_var_ = mapping.lane_var; // analysis placeholder; never escapes
    chunk_analysis_ = mapping.chunk_var;

    // The substitution loop_var -> chunk_analysis_ * L + lane_var_ drives
    // all address/index rewriting (same expression the shared helper
    // uses).
    lane_indexed_expr_ = mapping.index_expr;

    // A divisible extent uses the caller's shared full mask (constant
    // across chunks). A non-divisible extent gets a per-unit scalar
    // `remaining` that starts at the logical extent E and decreases by L
    // per chunk; each chunk binds a dynamic create_mask(remaining, L)
    // before the body and updates remaining at the chunk end. The tail
    // chunk's mask zeroes the lanes past E, which protects the store
    // (loads still read the full padded range — the caller guarantees
    // its readability; the store mask never widens a write past E).
    // The mask var must exist before the body conversion: the
    // converted vstore/vadd reference it while the body is built.
    const bool has_tail = mapping.tail_lanes != 0;
    if (!has_tail) {
      mask_var_ = shared_mask_;
    } else {
      // decl_buffer shapes the Var with a local.var pointer type; the
      // codegen reads the scope from there and emits the scalar
      // local-variable form the PTODSL surface expects.
      remaining_buffer_ =
          tirx::decl_buffer({IntImm(DataType::Int(32), 1)}, DataType::Int(32),
                            "remaining", "local.var");
      // The per-chunk mask bind is emitted inside the chunk loop below;
      // this var is what it binds and what the body references.
      mask_var_ = Var("mask", DataType::Bool(static_cast<int>(lanes_)));
    }

    // Loop shells: analyze the unit body for preserved serial loops
    // before conversion — set up carry promotion where the canonical
    // pattern matches, and reject order-unsafe shapes.
    AnalyzeLoopShell(
        inner != nullptr ? static_cast<const Stmt &>(inner->body)
                         : static_cast<const Stmt &>(loop->body),
        region_body);

    // Convert the unit body statements in order. For a 2D unit this is
    // the innermost compute body; the outer (kept) coordinate is
    // re-emitted as a serial loop around the chunk loop.
    Array<Stmt> unit_stmts;
    ConvertBody(inner != nullptr ? static_cast<const Stmt &>(inner->body)
                                 : static_cast<const Stmt &>(loop->body),
                &unit_stmts);

    // Drop binds whose value is never used (address-only aliases and
    // unused source binds strand vci/vbrc/vadd/vload chains) before the
    // chunk loop is built.
    Stmt eliminated = DeadBindElim()(SeqStmt::Flatten(unit_stmts));
    Array<Stmt> chunk_seq;
    if (!has_tail) {
      chunk_seq.push_back(std::move(eliminated));
    } else {
      // Per-chunk mask bind + body + remaining update, in that order.
      // mask_var_ was created before the body conversion.
      PrimExpr remaining_load =
          BufferLoad(remaining_buffer_, {make_zero(DataType::Int(32))});
      PrimExpr mask_value =
          Call(DataType::Bool(static_cast<int>(lanes_)), VmiOp("create_mask"),
               {remaining_load},
               {{"size", IntImm(DataType::Int(32), static_cast<int>(lanes_))}});
      chunk_seq.push_back(Bind(mask_var_, mask_value));
      chunk_seq.push_back(std::move(eliminated));
      PrimExpr next_remaining =
          remaining_load -
          IntImm(DataType::Int(32), static_cast<int64_t>(lanes_));
      chunk_seq.push_back(BufferStore(remaining_buffer_, next_remaining,
                                      {make_zero(DataType::Int(32))}));
    }
    Stmt chunk_body = SeqStmt::Flatten(chunk_seq);
    Stmt chunk_loop =
        For(chunk_var_, make_zero(chunk_var_->dtype),
            IntImm(chunk_var_->dtype, static_cast<int64_t>(nchunks_)),
            ForKind::kSerial, chunk_body);
    if (inner == nullptr) {
      if (!has_tail) {
        return WrapCarryAlloc(chunk_loop);
      }
      // remaining lives across the whole chunk loop: declare it, set it
      // to E once before the loop.
      Stmt init =
          BufferStore(remaining_buffer_,
                      IntImm(DataType::Int(32),
                             static_cast<int64_t>(mapping.logical_extent)),
                      {make_zero(DataType::Int(32))});
      return WrapCarryAlloc(SeqStmt::Flatten(
          AllocBuffer(remaining_buffer_), SeqStmt::Flatten(init, chunk_loop)));
    }
    // 2D: keep the non-vectorized coordinate as the outer
    // serial loop; the vectorized dimension becomes the chunk loop.
    const Var kept_var =
        mapping.select_inner ? loop->loop_var : inner->loop_var;
    const int64_t kept_extent =
        mapping.select_inner ? mapping.outer_extent : mapping.inner_extent;
    Stmt kept_body = chunk_loop;
    if (has_tail) {
      // remaining resets to E inside the kept loop: every row starts a
      // fresh recursion, so rows are independent and the tail chunk of
      // each row masks exactly that row's tail lanes.
      Stmt init =
          BufferStore(remaining_buffer_,
                      IntImm(DataType::Int(32),
                             static_cast<int64_t>(mapping.logical_extent)),
                      {make_zero(DataType::Int(32))});
      kept_body = SeqStmt::Flatten(init, chunk_loop);
    }
    Stmt kept_loop =
        For(kept_var, make_zero(kept_var->dtype),
            IntImm(kept_var->dtype, static_cast<int64_t>(kept_extent)),
            ForKind::kSerial, kept_body);
    if (!has_tail) {
      return WrapCarryAlloc(kept_loop);
    }
    return WrapCarryAlloc(
        SeqStmt::Flatten(AllocBuffer(remaining_buffer_), kept_loop));
  }

  // Substitute the analysis chunk var with the real loop var q (and keep
  // the lane placeholder for lane-binding steps).
  PrimExpr BindChunkToQ(const PrimExpr &expr) const {
    class Replacer : public ExprMutator {
    public:
      Replacer(const VarNode *from, const Var &to) : from_(from), to_(to) {}
      PrimExpr VisitExpr_(const VarNode *op) final {
        if (op == from_) {
          return to_;
        }
        return ffi::GetRef<PrimExpr>(op);
      }

    private:
      const VarNode *from_;
      Var to_;
    };
    return Replacer(chunk_analysis_.get(), chunk_var_)(expr);
  }

  // Rewrite Var references inside metadata objects through value_map_
  // Follows containers (Array/Map) structurally; PrimExprs get a
  // var substitution.
  ObjectRef RemapMetadataVars(const ObjectRef &obj) const {
    if (!obj.defined()) {
      return obj;
    }
    if (const auto *expr = obj.as<PrimExprNode>()) {
      class RemapVars : public ExprMutator {
      public:
        RemapVars(const Map<Var, Var> &vmap) : vmap_(vmap) {}
        PrimExpr VisitExpr_(const VarNode *op) final {
          auto it = vmap_.find(GetRef<Var>(op));
          if (it != vmap_.end()) {
            return (*it).second;
          }
          return GetRef<PrimExpr>(op);
        }

      private:
        const Map<Var, Var> &vmap_;
      };
      return RemapVars(value_map_)(GetRef<PrimExpr>(expr));
    }
    // Containers are rejected by Verify (metadata must be a PrimExpr);
    // reaching here with one is an internal error.
    ICHECK(obj.as<PrimExprNode>() == nullptr)
        << "[VectorizeParallelToPTO] internal error: container metadata "
           "passed Verify";
    return obj;
  }

  PrimExpr SubstituteLaneZero(const PrimExpr &expr) const {
    class ZeroLane : public ExprMutator {
    public:
      ZeroLane(const Var &lane) : lane_(lane) {}
      PrimExpr VisitExpr_(const VarNode *op) final {
        if (op == lane_.get()) {
          return make_zero(lane_->dtype);
        }
        return ffi::GetRef<PrimExpr>(op);
      }

    private:
      Var lane_;
    };
    return ZeroLane(lane_var_)(expr);
  }

  // ------------------------------------------------------------------
  // Statement conversion (pre-binds first, then the statement)
  // ------------------------------------------------------------------
  void ConvertBody(const Stmt &body, Array<Stmt> *out) {
    if (const auto *seq = body.as<SeqStmtNode>()) {
      for (const auto &s : seq->seq) {
        ConvertBody(s, out);
      }
      return;
    }
    if (const auto *bind = body.as<BindNode>()) {
      // Whitelist is Verify's job; defensive check here.
      Array<Stmt> pre;
      PrimExpr value = ConvertExpr(bind->value, &pre);
      // Pre-binds first, then the Bind itself.
      for (const auto &s : pre) {
        out->push_back(s);
      }
      Var new_var(bind->var->name_hint, value.dtype());
      bind_env_.Set(bind->var, bind->value); // analysis environment
      value_map_.Set(bind->var, new_var);    // value environment
      out->push_back(Bind(new_var, value));
      return;
    }
    if (const auto *store = body.as<BufferStoreNode>()) {
      if (carry_seed_ != nullptr && store == carry_seed_) {
        // Seed of the promoted carry: forward the value into the vreg
        // buffer instead of storing to UB, so the shell's loads never
        // see a round-trip.
        Array<Stmt> pre;
        PrimExpr value = ConvertExpr(store->value, &pre);
        if (value.dtype().lanes() == 1) {
          value = EmitBind(
              "vbrc_val",
              Call(VectorDType(value.dtype(), lanes_), VmiOp("vbrc"), {value},
                   {{"size",
                     IntImm(DataType::Int(32), static_cast<int>(lanes_))}}),
              &pre);
        }
        for (const auto &s : pre) {
          out->push_back(s);
        }
        out->push_back(BufferStore(carry_vreg_, value,
                                   {make_zero(DataType::Int(32))}));
        return;
      }
      if (carry_active_ && IsCarryAccess(store->buffer, store->indices)) {
        // Carry update inside the shell: write the vreg buffer instead
        // of a vstore. Unwrap a trailing pure Bind so the emitted
        // assignment keeps the direct `acc = f(acc, ...)` form the
        // PTODSL rewriter turns into scf.for iter_args.
        Array<Stmt> pre;
        PrimExpr value = ConvertExpr(store->value, &pre);
        if (const auto *var = value.as<VarNode>()) {
          if (!pre.empty()) {
            const Stmt &last = pre.back();
            if (const auto *b = last.as<BindNode>()) {
              if (b->var.get() == var) {
                Array<Stmt> kept;
                for (size_t i = 0; i + 1 < pre.size(); ++i) {
                  kept.push_back(pre[i]);
                }
                pre = kept;
                value = b->value;
              }
            }
          }
        }
        for (const auto &s : pre) {
          out->push_back(s);
        }
        out->push_back(BufferStore(carry_vreg_, value,
                                   {make_zero(DataType::Int(32))}));
        return;
      }
      Array<Stmt> pre;
      PrimExpr value = ConvertExpr(store->value, &pre);
      auto access = pto_analysis::AnalyzeBufferAccess(
          store->buffer, store->indices, /*is_write=*/true, loop_var_,
          CurrentMapping(), &analyzer_, bind_env_);
      if (access.pattern == pto_analysis::AccessPattern::kContinuous) {
        if (value.dtype().lanes() == 1) {
          value = EmitBind(
              "vbrc_val",
              Call(VectorDType(value.dtype(), lanes_), VmiOp("vbrc"), {value},
                   {{"size",
                     IntImm(DataType::Int(32), static_cast<int>(lanes_))}}),
              &pre);
        }
        Array<PrimExpr> start_idx;
        for (const auto &idx : access.start_indices) {
          start_idx.push_back(
              analyzer_.Simplify(BindChunkToQ(SubstituteLaneZero(idx))));
        }
        PrimExpr ptr =
            Call(DataType::Handle(), tl::access_ptr(),
                 {BufferLoad(store->buffer, start_idx),
                  IntImm(DataType::Int(32), static_cast<int>(lanes_)),
                  IntImm(DataType::Int(32), 2)});
        for (const auto &s : pre) {
          out->push_back(s);
        }
        out->push_back(Evaluate(
            Call(DataType::Void(), VmiOp("vstore"),
                 {value, ptr, IntImm(DataType::Int(32), 0), mask_var_})));
      } else {
        // Verify rejects lane-uniform stores (the current
        // support boundary), so this branch is unreachable. Treat any
        // survivor as an internal error, same style as the wrapper
        // checks above.
        ICHECK(false)
            << "[VectorizeParallelToPTO] internal error: lane-uniform store `"
            << store->buffer->name
            << "` passed Verify; lane-uniform stores must be rejected by "
               "VerifyParallelToPTO";
      }
      return;
    }
    if (const auto *attr = body.as<AttrStmtNode>()) {
      // Verify guarantees the metadata has no lane-dependent semantics
      // (direct or via bind aliases). Rewrite its Var references through
      // value_map_ so already-converted binds are not left dangling
      // The body is converted underneath.
      Array<Stmt> stmts;
      // The body is a nested statement scope — keep the
      // source-bind and value maps scoped with it, so a bind defined
      // inside cannot remap a later sibling's reference (Verify rejects
      // that input; this keeps the transformation state consistent with
      // the unit visitor's scope handling).
      Map<Var, PrimExpr> saved_env = bind_env_;
      Map<Var, Var> saved_map = value_map_;
      ConvertBody(attr->body, &stmts);
      bind_env_ = std::move(saved_env);
      value_map_ = std::move(saved_map);
      Stmt inner = SeqStmt::Flatten(stmts);
      // Default to the *original* node. Only PrimExpr metadata is
      // remapped; a non-object Any (int/str atom metadata) must forward
      // unchanged — the old code wrote back a default-constructed (empty)
      // node when the try_cast failed.
      ffi::Any node = attr->node;
      if (auto ne = attr->node.try_cast<ObjectRef>()) {
        if (const auto *pe = ne.value().as<PrimExprNode>()) {
          node = RemapMetadataVars(GetRef<PrimExpr>(pe));
        }
      }
      PrimExpr value = attr->value;
      if (const auto *ve = attr->value.as<PrimExprNode>()) {
        value = RemapMetadataVars(GetRef<PrimExpr>(ve)).as<PrimExpr>().value();
      }
      out->push_back(AttrStmt(node, attr->attr_key, value, inner));
      return;
    }
    if (const auto *eval = body.as<EvaluateNode>()) {
      // Convert the expression with the same machinery so no stale loop
      // variable survives; the whitelist is Verify's job.
      Array<Stmt> pre;
      PrimExpr value = ConvertExpr(eval->value, &pre);
      for (const auto &s : pre) {
        out->push_back(s);
      }
      out->push_back(Evaluate(value));
      return;
    }
    if (const auto *loop = body.as<ForNode>()) {
      ICHECK(loop->kind == ForKind::kSerial)
          << "[VectorizeParallelToPTO] internal error: non-serial loop "
             "passed Verify";
      // Loop-shell preservation: a serial loop admitted by Verify is
      // copied through as a scalar shell; the body is converted with the
      // same statement machinery. Binds defined inside stay scoped to
      // the shell (same discipline as the AttrStmt branch). Verify has
      // pinned min/extent to constants and the step to 1, so the shell
      // is rebuilt with default step semantics.
      Array<Stmt> inner;
      Map<Var, PrimExpr> saved_env = bind_env_;
      Map<Var, Var> saved_map = value_map_;
      bool saved_carry = carry_active_;
      if (carry_loop_ != nullptr && loop == carry_loop_) {
        carry_active_ = true;
      }
      ConvertBody(loop->body, &inner);
      carry_active_ = saved_carry;
      bind_env_ = std::move(saved_env);
      value_map_ = std::move(saved_map);
      out->push_back(For(loop->loop_var, loop->min, loop->extent,
                         ForKind::kSerial, SeqStmt::Flatten(inner)));
      if (carry_loop_ != nullptr && loop == carry_loop_) {
        out->push_back(EmitCarryWriteback());
      }
      return;
    }
    LOG(FATAL) << "[VectorizeParallelToPTO] internal error: unsupported "
                  "statement passed Verify: "
               << body->GetTypeKey();
  }

  pto_analysis::PtoInverseMapping CurrentMapping() const {
    // Return the *complete* inverse mapping saved for this unit.
    // Rebuilding a 1D-only mapping here dropped is_2d, the two loop vars,
    // their extents and select_inner, so the shared access proof lost the
    // kept coordinate's range and Vectorize rejected units Verify had
    // accepted. The analysis placeholders (chunk/lane vars) are exactly
    // the ones saved from InvertPtoLaneLayout; only the emitted chunk
    // loop var q is separate (see BindChunkToQ).
    return unit_mapping_;
  }

  // ------------------------------------------------------------------
  // Expression conversion (index vars resolve through bind_env_)
  // ------------------------------------------------------------------
  PrimExpr ConvertExpr(const PrimExpr &expr, Array<Stmt> *pre) {
    if (const auto *load = expr.as<BufferLoadNode>()) {
      if (carry_active_ && IsCarryAccess(load->buffer, load->indices)) {
        // Promoted carry read: the value lives in the vreg buffer.
        return BufferLoad(carry_vreg_, {make_zero(DataType::Int(32))});
      }
      auto access = pto_analysis::AnalyzeBufferAccess(
          load->buffer, load->indices, /*is_write=*/false, loop_var_,
          CurrentMapping(), &analyzer_, bind_env_);
      if (access.pattern == pto_analysis::AccessPattern::kContinuous) {
        Array<PrimExpr> start_idx;
        for (const auto &idx : access.start_indices) {
          start_idx.push_back(
              analyzer_.Simplify(BindChunkToQ(SubstituteLaneZero(idx))));
        }
        PrimExpr ptr =
            Call(DataType::Handle(), tl::access_ptr(),
                 {BufferLoad(load->buffer, start_idx),
                  IntImm(DataType::Int(32), static_cast<int>(lanes_)),
                  IntImm(DataType::Int(32), 1)});
        // VMI with a return value goes through EmitBind.
        return EmitBind("vload",
                        Call(VectorDType(load->dtype, lanes_), VmiOp("vload"),
                             {ptr, IntImm(DataType::Int(32), 0)},
                             {{"size", IntImm(DataType::Int(32),
                                              static_cast<int>(lanes_))}}),
                        pre);
      }
      // lane-uniform scalar load: keep the BufferLoad (lane bound to 0,
      // chunk bound to q — indices resolve through the bind env
      // inside the shared helper already) and bind it to a scalar Var in
      // the current chunk; later uses (vbrc broadcast, scalar arithmetic,
      // store values) reference that Var. Each textual occurrence binds
      // separately — no cross-access load reuse. The Var dtype is the
      // buffer's own scalar dtype.
      Array<PrimExpr> indices;
      for (const auto &idx : access.indices_after_inverse) {
        indices.push_back(
            analyzer_.Simplify(BindChunkToQ(SubstituteLaneZero(idx))));
      }
      return EmitBind("sload", BufferLoad(load->buffer, indices), pre);
    }
    if (const auto *var = expr.as<VarNode>()) {
      auto it = value_map_.find(GetRef<Var>(var));
      if (it != value_map_.end()) {
        return (*it).second;
      }
      if (var == loop_var_.get()) {
        // lane_indexed used as a value: explicit vci vector.
        PrimExpr base = analyzer_.Simplify(
            BindChunkToQ(SubstituteLaneZero(lane_indexed_expr_)));
        return EmitBind(
            "vci",
            Call(VectorDType(var->dtype, lanes_), VmiOp("vci"), {base},
                 {{"size",
                   IntImm(DataType::Int(32), static_cast<int>(lanes_))}}),
            pre);
      }
      return GetRef<PrimExpr>(var);
    }
    if (expr.as<IntImmNode>() || expr.as<FloatImmNode>()) {
      return expr;
    }
    if (const auto *add = expr.as<AddNode>()) {
      return ConvertBinary(add->a, add->b, "vadd", pre);
    }
    if (const auto *sub = expr.as<SubNode>()) {
      return ConvertBinary(sub->a, sub->b, "vsub", pre);
    }
    if (const auto *mul = expr.as<MulNode>()) {
      return ConvertBinary(mul->a, mul->b, "vmul", pre);
    }
    if (const auto *div = expr.as<DivNode>()) {
      return ConvertBinary(div->a, div->b, "vdiv", pre);
    }
    if (const auto *call = expr.as<CallNode>()) {
      // M2: lane-mask boolean combinations and the topk_gate infinity constant
      // come first. They are not unary math, so the strict kUnaryMap check
      // below must not see them (that table only covers fabs/exp/log/sqrt).
      std::string name;
      if (const auto *op_node = call->op.as<OpNode>()) {
        name = op_node->name;
      }
      if (name == "tl.infinity" || name == "tir.infinity") {
        // Lane-uniform +inf scalar (frontend -T.infinity is Mul(inf, -1)).
        // Keep the Call; vbrc / Mul / Select consume it like a FloatImm.
        return GetRef<PrimExpr>(call);
      }
      if (name == "tirx.bitwise_and" || name == "tir.bitwise_and" ||
          name == "tir.And") {
        ICHECK_EQ(call->args.size(), 2);
        return ConvertMaskLogic(call->args[0], call->args[1], "mask_and", pre);
      }
      if (name == "tirx.bitwise_or" || name == "tir.bitwise_or" ||
          name == "tir.Or") {
        ICHECK_EQ(call->args.size(), 2);
        return ConvertMaskLogic(call->args[0], call->args[1], "mask_or", pre);
      }
      if (name == "tirx.bitwise_not" || name == "tir.bitwise_not" ||
          name == "tir.Not") {
        ICHECK_EQ(call->args.size(), 1);
        PrimExpr a = ConvertExpr(call->args[0], pre);
        DataType mask_ty = DataType::Bool(static_cast<int>(lanes_));
        return EmitBind("mnot", Call(mask_ty, VmiOp("mask_not"), {a}), pre);
      }
      // Hardware-backed unary math (fabs/exp/log/sqrt); the dtype matrix
      // mirrors the ISA documentation (see IsSupportedVectorUnaryOp).
      static const std::unordered_map<std::string, std::string> kUnaryMap = {
          {"tirx.fabs", "vabs"},
          {"tirx.exp", "vexp"},
          {"tirx.log", "vln"},
          {"tirx.sqrt", "vsqrt"}};
      auto call_op = call->op.as<Op>();
      ICHECK(call_op.has_value())
          << "[VectorizeParallelToPTO] internal error: non-Op call passed "
             "Verify: "
          << expr;
      auto it = kUnaryMap.find(call_op.value()->name);
      ICHECK(it != kUnaryMap.end())
          << "[VectorizeParallelToPTO] internal error: call passed Verify "
             "without a VMI mapping: "
          << call_op.value()->name;
      PrimExpr a = ConvertExpr(call->args[0], pre);
      if (a.dtype().lanes() == 1) {
        return Call(expr.dtype(), call->op, {a});
      }
      return EmitBind(it->second,
                      Call(VectorDType(expr.dtype().element_of(), lanes_),
                           VmiOp(it->second), {a, mask_var_}),
                      pre);
    }
    if (const auto *cast = expr.as<CastNode>()) {
      // Convert the input exactly once, then plan from the explicit shape of
      // that converted input (the original Cast carries the element types).
      PrimExpr value = ConvertExpr(cast->value, pre);
      pto_analysis::CastContext context;
      if (value.dtype().lanes() == 1) {
        context.value_kind = pto_analysis::CastValueKind::kScalar;
        context.logical_lanes = 1;
      } else {
        ICHECK_EQ(value.dtype().lanes(), lanes_)
            << "[VectorizeParallelToPTO] internal error: Cast input has "
               "unexpected lanes "
            << value.dtype().lanes() << " (current VL " << lanes_ << ")";
        context.value_kind = pto_analysis::CastValueKind::kVectorVmi;
        context.logical_lanes = lanes_;
      }
      ICHECK_EQ(value.dtype().element_of(), cast->value.dtype())
          << "[VectorizeParallelToPTO] internal error: converted Cast input "
             "element type "
          << value.dtype().element_of() << " does not match the original "
          << cast->value.dtype();
      std::string plan_why;
      auto plan = pto_analysis::PlanValueCast(cast, context, &plan_why);
      ICHECK(plan.has_value())
          << "[VectorizeParallelToPTO] internal error: Cast passed Verify "
             "without a conversion plan: "
          << plan_why;
      if (const auto *scalar =
              std::get_if<pto_analysis::ScalarCastPlan>(&plan.value())) {
        ICHECK_EQ(value.dtype().lanes(), 1)
            << "[VectorizeParallelToPTO] internal error: scalar Cast plan "
               "for a vector input";
        if (scalar->action == pto_analysis::ScalarCastAction::kKeepInput) {
          return value;
        }
        // Rebuild the ordinary scalar Cast with the validated annotations and
        // the original span; the scalar lowering itself is unchanged.
        return Cast(cast->dtype, value, scalar->annotations, cast->span);
      }
      const auto &vector = std::get<pto_analysis::VectorCastPlan>(plan.value());
      PrimExpr current = value;
      for (const auto &step : vector.steps) {
        ICHECK_EQ(current.dtype().element_of(), step.from)
            << "[VectorizeParallelToPTO] internal error: Cast step expects "
            << step.from << ", got " << current.dtype().element_of();
        Map<String, ObjectRef> attrs;
        attrs.Set("to_dtype", StringImm(TvmDtypeName(step.to)));
        PrimExpr call;
        if (step.op == pto_analysis::VectorCastOp::kIntegerReinterpret) {
          call = Call(VectorDType(step.to, lanes_), VmiOp("vinterpret_cast"),
                      {current}, attrs);
        } else {
          if (!step.saturate.empty()) {
            attrs.Set("saturate", StringImm(step.saturate));
          }
          if (!step.rounding.empty()) {
            attrs.Set("rounding", StringImm(step.rounding));
          }
          call = Call(VectorDType(step.to, lanes_), VmiOp("vcvt"), {current},
                      attrs);
        }
        current = EmitBind("cast", call, pre);
      }
      // Zero steps is the proven identity; a final type check keeps every
      // multi-step chain honest (source convert once, dtype/VL continuous).
      ICHECK_EQ(current.dtype(), VectorDType(cast->dtype, lanes_))
          << "[VectorizeParallelToPTO] internal error: Cast plan ended at "
          << current.dtype() << ", expected "
          << VectorDType(cast->dtype, lanes_);
      return current;
    }
    if (const auto *andn = expr.as<AndNode>()) {
      return ConvertMaskLogic(andn->a, andn->b, "mask_and", pre);
    }
    if (const auto *orn = expr.as<OrNode>()) {
      return ConvertMaskLogic(orn->a, orn->b, "mask_or", pre);
    }
    if (const auto *notn = expr.as<NotNode>()) {
      PrimExpr a = ConvertExpr(notn->a, pre);
      DataType mask_ty = DataType::Bool(static_cast<int>(lanes_));
      ICHECK(a.dtype().lanes() > 1)
          << "[VectorizeParallelToPTO] mask_not expects a lane mask";
      return EmitBind("mnot", Call(mask_ty, VmiOp("mask_not"), {a}), pre);
    }
    if (const auto *sel = expr.as<SelectNode>()) {

      PrimExpr cond = ConvertExpr(sel->condition, pre);
      PrimExpr tval = ConvertExpr(sel->true_value, pre);
      PrimExpr fval = ConvertExpr(sel->false_value, pre);
      if (tval.dtype().lanes() == 1) {
        tval = EmitBind(
            "brc",
            Call(VectorDType(tval.dtype(), lanes_), VmiOp("vbrc"), {tval},
                 {{"size",
                   IntImm(DataType::Int(32), static_cast<int>(lanes_))}}),
            pre);
      }
      if (fval.dtype().lanes() == 1) {
        fval = EmitBind(
            "brc",
            Call(VectorDType(fval.dtype(), lanes_), VmiOp("vbrc"), {fval},
                 {{"size",
                   IntImm(DataType::Int(32), static_cast<int>(lanes_))}}),
            pre);
      }
      ICHECK(cond.dtype().lanes() > 1)
          << "[VectorizeParallelToPTO] Select condition must be a lane mask "
             "after conversion (got scalar); Phase 1 expects lane-varying "
             "predicates from SimdVFLowerControlFlow";
      return EmitBind(
          "vsel", Call(tval.dtype(), VmiOp("vsel"), {cond, tval, fval}), pre);
    }
    if (const auto *eq = expr.as<EQNode>()) {
      return ConvertCompare(eq->a, eq->b, "eq", pre);
    }
    if (const auto *ne = expr.as<NENode>()) {
      return ConvertCompare(ne->a, ne->b, "ne", pre);
    }
    if (const auto *lt = expr.as<LTNode>()) {
      return ConvertCompare(lt->a, lt->b, "lt", pre);
    }
    if (const auto *le = expr.as<LENode>()) {
      return ConvertCompare(le->a, le->b, "le", pre);
    }
    if (const auto *gt = expr.as<GTNode>()) {
      return ConvertCompare(gt->a, gt->b, "gt", pre);
    }
    if (const auto *ge = expr.as<GENode>()) {
      return ConvertCompare(ge->a, ge->b, "ge", pre);
    }
    LOG(FATAL) << "[VectorizeParallelToPTO] internal error: unsupported "
                  "expression passed Verify: "
               << expr->GetTypeKey();
  }

  PrimExpr ConvertCompare(const PrimExpr &lhs, const PrimExpr &rhs,
                          const char *cmp, Array<Stmt> *pre) {
    PrimExpr a = ConvertExpr(lhs, pre);
    PrimExpr b = ConvertExpr(rhs, pre);
    bool a_vec = a.dtype().lanes() > 1;
    bool b_vec = b.dtype().lanes() > 1;
    DataType mask_ty = DataType::Bool(static_cast<int>(lanes_));
    if (!a_vec && !b_vec) {
      // Lane-uniform compare: keep scalar relational (Select of scalars is
      // rare inside Parallel; Verify still allows it).
      if (std::string(cmp) == "eq")
        return a == b;
      if (std::string(cmp) == "ne")
        return a != b;
      if (std::string(cmp) == "lt")
        return a < b;
      if (std::string(cmp) == "le")
        return a <= b;
      if (std::string(cmp) == "gt")
        return a > b;
      if (std::string(cmp) == "ge")
        return a >= b;
      LOG(FATAL) << "[VectorizeParallelToPTO] unknown cmp " << cmp;
    }
    if (a_vec && !b_vec) {
      return EmitBind(
          "vcmps",
          Call(mask_ty, VmiOp("vcmps"), {a, b, mask_var_, StringImm(cmp)}),
          pre);
    }
    if (!a_vec && b_vec) {
      a = EmitBind(
          "brc",
          Call(VectorDType(a.dtype(), lanes_), VmiOp("vbrc"), {a},
               {{"size", IntImm(DataType::Int(32), static_cast<int>(lanes_))}}),
          pre);
    }
    return EmitBind(
        "vcmp", Call(mask_ty, VmiOp("vcmp"), {a, b, mask_var_, StringImm(cmp)}),
        pre);
  }

  PrimExpr ConvertMaskLogic(const PrimExpr &lhs, const PrimExpr &rhs,
                            const char *op_name, Array<Stmt> *pre) {
    PrimExpr a = ConvertExpr(lhs, pre);
    PrimExpr b = ConvertExpr(rhs, pre);
    DataType mask_ty = DataType::Bool(static_cast<int>(lanes_));
    ICHECK(a.dtype().lanes() > 1 && b.dtype().lanes() > 1)
        << "[VectorizeParallelToPTO] " << op_name
        << " expects lane masks (Phase 2)";
    return EmitBind(op_name, Call(mask_ty, VmiOp(op_name), {a, b}), pre);
  }

  PrimExpr ConvertBinary(const PrimExpr &lhs, const PrimExpr &rhs,
                         const std::string &vmi_name, Array<Stmt> *pre) {
    PrimExpr a = ConvertExpr(lhs, pre);
    PrimExpr b = ConvertExpr(rhs, pre);
    bool a_vec = a.dtype().lanes() > 1;
    bool b_vec = b.dtype().lanes() > 1;
    if (!a_vec && !b_vec) {
      // Lane-uniform folds stay scalar with the TIR semantics of the node.
      if (vmi_name == "vadd") {
        return a + b;
      }
      if (vmi_name == "vsub") {
        return a - b;
      }
      if (vmi_name == "vmul") {
        return a * b;
      }
      if (vmi_name == "vdiv") {
        return div(a, b);
      }
      ICHECK(false) << "[VectorizeParallelToPTO] internal error: unknown "
                       "binary op "
                    << vmi_name;
    }
    if (a_vec && !b_vec) {
      b = EmitBind(
          "brc",
          Call(VectorDType(b.dtype(), lanes_), VmiOp("vbrc"), {b},
               {{"size", IntImm(DataType::Int(32), static_cast<int>(lanes_))}}),
          pre);
    } else if (!a_vec && b_vec) {
      a = EmitBind(
          "brc",
          Call(VectorDType(a.dtype(), lanes_), VmiOp("vbrc"), {a},
               {{"size", IntImm(DataType::Int(32), static_cast<int>(lanes_))}}),
          pre);
    }
    return EmitBind(vmi_name,
                    Call(VectorDType(a.dtype().element_of(), lanes_),
                         VmiOp(vmi_name), {a, b, mask_var_}),
                    pre);
  }

  // TVM dtype name identical to Python str(dtype) (vmi.py ABI):
  // float32 -> "float32", bfloat16 -> "bfloat16", int8 -> "int8", ...
  static std::string TvmDtypeName(DataType dtype) {
    std::ostringstream os;
    if (dtype.is_bfloat16()) {
      return "bfloat16";
    }
    // The FP8 spellings the PTO dtype parser recognizes; both are the TIR
    // dtype names, not the PTO target names.
    if (dtype.is_float8_e4m3fn()) {
      return "float8_e4m3fn";
    }
    if (dtype.is_float8_e5m2()) {
      return "float8_e5m2";
    }
    if (dtype.is_float()) {
      os << "float" << dtype.bits();
    } else if (dtype.is_int()) {
      os << "int" << dtype.bits();
    } else if (dtype.is_uint()) {
      os << "uint" << dtype.bits();
    } else if (dtype.is_bool()) {
      return "bool";
    } else {
      os << dtype;
    }
    return os.str();
  }

  Var EmitBind(const std::string &name, const PrimExpr &value,
               Array<Stmt> *pre) {
    Var result(name + "_" + std::to_string(tmp_counter_++), value.dtype());
    pre->push_back(Bind(result, value));
    return result;
  }

  arith::Analyzer analyzer_;
  Var shared_mask_; // one full mask per VF, shared by all units

  // Per-unit state.
  Var loop_var_;
  Var chunk_var_;      // real emitted loop var (q)
  Var chunk_analysis_; // analysis placeholder from the inverse mapping
  Var lane_var_;       // analysis placeholder; never escapes
  PrimExpr lane_indexed_expr_;
  /*! The complete inverse mapping of the unit being converted
   * (2D fields included), consumed by CurrentMapping(). */
  pto_analysis::PtoInverseMapping unit_mapping_;
  int64_t lanes_ = 0;
  int64_t nchunks_ = 0;
  Var mask_var_;
  /*! Per-unit scalar buffer holding `remaining` for a non-divisible
   * extent (invalid for divisible units). */
  tirx::Buffer remaining_buffer_{};
  // Loop-shell carry promotion (AnalyzeLoopShell): when a unit's
  // preserved serial shell matches the canonical carry pattern, the
  // carried buffer's in-shell accesses are rewritten to this local.var
  // vreg buffer, so the loop never round-trips through UB — the
  // round-trip is not order-safe on the current PTOAS (no automatic
  // mem_bar between the carried vstore and the next iteration's vload).
  const BufferStoreNode *carry_seed_ = nullptr;
  const ForNode *carry_loop_ = nullptr;
  tirx::Buffer carry_src_{};
  tirx::Buffer carry_vreg_{};
  Array<PrimExpr> carry_indices_{};
  bool carry_active_ = false;
  int tmp_counter_ = 0;
  Map<Var, PrimExpr> bind_env_; // analysis env: original var -> def expr
  Map<Var, Var> value_map_;     // value env: original var -> converted var
};

} // namespace

namespace transform {

tvm::transform::Pass VectorizeParallelToPTO() {
  auto pass_func = [](PrimFunc f, const IRModule &m,
                      const tvm::transform::PassContext &ctx) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsPTO(target.value())) {
      return f;
    }
    return VectorizeParallelToPTOImpl::Substitute(std::move(f));
  };
  return tvm::tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.VectorizeParallelToPTO", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.VectorizeParallelToPTO",
                        VectorizeParallelToPTO);
}

} // namespace transform

} // namespace tl
} // namespace tvm
