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
 *   2. build one chunk loop q in [0, ceil(E/L)), mapping the logical
 *      index to q * L + lane;
 *   3. classify every access via the shared MemoryAccess analysis
 *      (continuous -> vload/vstore via tl.access_ptr(start, L, rw);
 *      lane-uniform load -> scalar BufferLoad; indexed load/store and
 *      lane-uniform store -> masked gather/scatter with element offsets);
 *   4. convert value expressions (value roles tracked by result dtype);
 *      every VMI with a return value is bound through EmitBind before use;
 *   5. delete the consumed layout annotation, keep the SIMD_VF block
 *      annotations, and leave no unconverted Parallel behind.
 *
 * Supported shapes: 1D parallel units and two-layer adjacent Parallel nests
 * (2D — the nest converts as one unit with the layout
 * annotation on its outermost For); sequential units allowed; static
 * extents with full or tail masks. A third parallel layer or any other
 * unsupported structure is rejected by Verify.
 *
 * Non-divisible extents use a per-chunk mask in the same loop. Continuous
 * vload is unmasked: the caller must provide initialized readable padding
 * for every complete logical vector. Indexed offsets are sanitized in every
 * iteration when the unit has a tail, and passed directly otherwise.
 */

#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "buffer_version.h"
#include "parallel_to_pto_utils.h"
#include "pto_indexed_memory.h"

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
    // a Bind RHS and that is pure/read-only by construction — vload/vgather
    // (reads) and arithmetic, select or conversion ops (pure), enumerated from
    // the ConvertUnit/VmiOp call sites — or the pure tl.access_ptr address
    // descriptor nested inside a load operand (already whitelisted as a
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
        "tl.vmi.vload",  "tl.vmi.vgather", "tl.vmi.vsel",
        "tl.vmi.vbrc",   "tl.vmi.vci",     "tl.vmi.vadd",
        "tl.vmi.vsub",   "tl.vmi.vmul",    "tl.vmi.vdiv",
        "tl.vmi.vabs",   "tl.vmi.vexp",    "tl.vmi.vln",
        "tl.vmi.vsqrt",  "tl.vmi.vcvt",    "tl.vmi.vinterpret_cast",
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
    for (const Var &parameter : f->params) {
      mutator.parameters_.insert(parameter);
    }
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = mutator.VisitStmt(f->body);
    return GetRef<PrimFunc>(fptr);
  }

private:
  // AllocBuffer is a flat definition visible to later siblings. Every other
  // statement restores the owner environment on leaving its lexical scope.
  Stmt VisitStmt(const Stmt &stmt) final {
    if (stmt.as<AllocBufferNode>()) {
      return StmtExprMutator::VisitStmt(stmt);
    }
    const size_t allocation_mark = active_allocations_.size();
    Stmt result = StmtExprMutator::VisitStmt(stmt);
    active_allocations_.resize(allocation_mark);
    return result;
  }

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
    for (const Buffer &buffer : op->alloc_buffers) {
      active_allocations_.push_back(buffer);
    }
    if (op->name_hint != "SIMD_VF") {
      return StmtExprMutator::VisitStmt_(op);
    }
    // Shared region scan: the same implementation Verify uses, so
    // skip/reject/convert decisions cannot drift apart.
    const Stmt &region = op->body;
    pto_analysis::PtoRegionScan scan = pto_analysis::ScanPtoRegion(region);
    if (!scan.has_any_parallel && !scan.has_direct_reduce && !scan.has_fill) {
      // Pure hand-written VMI region (possibly with control flow): skipped
      // by both passes; no lanes annotation needed.
      return StmtExprMutator::VisitStmt_(op);
    }
    ICHECK(!scan.has_unsupported_wrapper)
        << "[VectorizeParallelToPTO] internal error: unsupported wrapper "
           "structure passed Verify: "
        << scan.issue;

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
      // Read the VF annotation directly: a reduce-only VF has no Parallel
      // layout from which to recover lanes. Verify checks layout consistency
      // when Parallel units are also present.
      auto lanes_annotation = op->annotations.Get("tl.simdvf_lanes");
      ICHECK(lanes_annotation.has_value());
      int64_t lanes = Downcast<IntImm>(lanes_annotation.value())->value;
      // Divisible units share a static full mask. Units with a tail use
      // their own per-iteration mask throughout the unified chunk loop.
      shared_mask_ = Var();
      bool needs_full_mask = false;
      for (const pto_analysis::PtoInverseMapping &mapping : mappings) {
        needs_full_mask |= mapping.tail_lanes == 0;
      }
      Stmt new_region = region;
      if (needs_full_mask) {
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
        new_region = ReplaceStmt(new_region, scan.units[i], mappings[i]);
      }
      int64_t saved_lanes = current_vf_lanes_;
      current_vf_lanes_ = lanes;
      new_body = VisitStmt(new_region);
      current_vf_lanes_ = saved_lanes;
    }

    pto_analysis::PtoRegionScan residual =
        pto_analysis::ScanPtoRegion(new_body);
    ICHECK(!residual.has_direct_reduce && !residual.has_fill)
        << "[VectorizeParallelToPTO] direct tileops "
           "must be standalone Evaluate(Call) statements";

    // new_body is (mask bind + region with all units converted); the
    // scope AttrStmt wrapper, if any, is inside it already.
    return SBlock(op->iter_vars, op->reads, op->writes, op->name_hint, new_body,
                  op->init, op->alloc_buffers, op->match_buffers,
                  op->annotations);
  }

  Stmt LowerDirectReduce(const Call &call, int64_t lanes) {
    pto_analysis::PtoDirectReducePlan plan =
        pto_analysis::PlanPtoDirectReduce(call, &analyzer_, buffer_versions_);
    bool absolute = plan.operation == "abssum" || plan.operation == "absmax";
    bool sum = plan.operation == "sum" || plan.operation == "abssum";
    bool maximum = plan.operation == "max" || plan.operation == "absmax";
    auto i32 = [](int64_t value) { return IntImm(DataType::Int(32), value); };
    DataType vector_dtype = plan.accumulator_dtype.with_lanes(lanes);
    DataType mask_dtype = DataType::Bool(lanes);
    Buffer accumulator =
        tirx::decl_buffer({i32(1)}, vector_dtype, "reduce_acc", "local.var");
    Var chunk("reduce_chunk", DataType::Int(32));
    Var output("reduce_output", DataType::Int(32));
    Var reduction("reduce_index", DataType::Int(32));
    Var full_mask("reduce_domain", mask_dtype);
    Var active_mask("reduce_active", mask_dtype);
    auto mask = [&](const PrimExpr &active) {
      return Call(mask_dtype, VmiOp("create_mask"), {active},
                  {{"size", i32(lanes)}});
    };
    auto pointer = [&](const BufferRegion &region,
                       const Array<PrimExpr> &indices, int64_t extent, int rw) {
      return Call(DataType::Handle(), tl::access_ptr(),
                  {BufferLoad(region->buffer, indices), i32(extent), i32(rw)});
    };
    auto source_indices = [&](const PrimExpr &out, const PrimExpr &red) {
      Array<PrimExpr> indices;
      for (const Range &range : plan.source->region) {
        indices.push_back(range->min);
      }
      indices.Set(plan.source_prefix + plan.reduction_axis, red);
      if (plan.logical_rank == 2) {
        indices.Set(plan.source_prefix + plan.output_axis, out);
      }
      return indices;
    };
    auto destination_indices = [&](const PrimExpr &out) {
      Array<PrimExpr> indices;
      for (const Range &range : plan.destination->region) {
        indices.push_back(range->min);
      }
      indices.Set(plan.destination_output_axis, out);
      return indices;
    };
    auto load = [&](const PrimExpr &out, const PrimExpr &red,
                    Array<Stmt> *statements) {
      DataType src_dtype = plan.source->buffer->dtype;
      int64_t footprint =
          ((lanes + 2048 / src_dtype.bits() - 1) / (2048 / src_dtype.bits())) *
          (2048 / src_dtype.bits());
      PrimExpr value = EmitBind(
          "reduce_load",
          Call(src_dtype.with_lanes(lanes), VmiOp("vload"),
               {pointer(plan.source, source_indices(out, red), footprint, 1),
                i32(0)},
               {{"size", i32(lanes)}}),
          statements);
      if (src_dtype != plan.accumulator_dtype) {
        value =
            EmitBind("reduce_cast",
                     Call(vector_dtype, VmiOp("vcvt"), {value},
                          {{"to_dtype",
                            StringImm(TvmDtypeName(plan.accumulator_dtype))}}),
                     statements);
      }
      if (absolute && !plan.accumulator_dtype.is_uint()) {
        value =
            EmitBind("reduce_abs", Call(vector_dtype, VmiOp("vabs"), {value}),
                     statements);
      }
      return value;
    };
    auto read_accumulator = [&]() { return BufferLoad(accumulator, {i32(0)}); };
    const char *fold_name = sum ? "vadd" : maximum ? "vmax" : "vmin";
    auto seed = [&](const PrimExpr &first, const PrimExpr &active,
                    Array<Stmt> *statements) -> PrimExpr {
      if (plan.operation != "absmax") {
        return first;
      }
      // Match the common Ascend/CUDA reducer's zero-initialized partial
      PrimExpr zero = EmitBind("reduce_zero",
                               Call(vector_dtype, VmiOp("vbrc"),
                                    {make_zero(plan.accumulator_dtype)},
                                    {{"size", i32(lanes)}}),
                               statements);
      return EmitBind("reduce_seed",
                      Call(vector_dtype, VmiOp("vmax"), {zero, first, active}),
                      statements);
    };
    Array<Stmt> body;
    if (plan.reduce_across_lanes) {
      PrimExpr first = load(output, i32(0), &body);
      first = seed(first, full_mask, &body);
      body.push_back(BufferStore(accumulator, first, {i32(0)}));
      auto accumulate = [&](const PrimExpr &red, const PrimExpr &active,
                            bool keep_inactive) {
        Array<Stmt> update;
        PrimExpr next = load(output, red, &update);
        PrimExpr folded = EmitBind("reduce_fold",
                                   Call(vector_dtype, VmiOp(fold_name),
                                        {read_accumulator(), next, active}),
                                   &update);
        if (keep_inactive) {
          folded = EmitBind("reduce_merge",
                            Call(vector_dtype, VmiOp("vsel"),
                                 {active, folded, read_accumulator()}),
                            &update);
        }
        update.push_back(BufferStore(accumulator, folded, {i32(0)}));
        return SeqStmt::Flatten(update);
      };
      int64_t full_chunks = plan.reduction_extent / lanes;
      int64_t tail_lanes = plan.reduction_extent % lanes;
      // The seed already covers the first chunk. Complete chunks need no
      // remaining-lane calculation or merge with the previous accumulator.
      if (full_chunks > 1) {
        body.push_back(For(chunk, i32(1), i32(full_chunks - 1),
                           ForKind::kSerial,
                           accumulate(chunk * i32(lanes), full_mask, false)));
      }
      if (full_chunks > 0 && tail_lanes != 0) {
        body.push_back(Bind(active_mask, mask(i32(tail_lanes))));
        body.push_back(accumulate(i32(full_chunks * lanes), active_mask, true));
      }
      ffi::Map<ffi::String, ObjectRef> annotations;
      if (sum && plan.accumulator_dtype.is_float()) {
        annotations.Set("reassoc", IntImm(DataType::Bool(), 1));
      }
      const char *reduce_name = sum ? "vcadd" : maximum ? "vcmax" : "vcmin";
      PrimExpr result =
          EmitBind("reduce_result",
                   Call(plan.accumulator_dtype, VmiOp(reduce_name),
                        {read_accumulator(), full_mask}, annotations),
                   &body);
      if (!plan.clear) {
        PrimExpr previous = EmitBind(
            "reduce_previous",
            Call(plan.accumulator_dtype, VmiOp("vload"),
                 {pointer(plan.destination, destination_indices(output), 1, 1),
                  i32(0)},
                 {{"size", i32(1)}}),
            &body);
        PrimExpr one = EmitBind("reduce_one",
                                Call(DataType::Bool(), VmiOp("create_mask"),
                                     {i32(1)}, {{"size", i32(1)}}),
                                &body);
        // Merge the previous output only after the cross-lane reduction;
        // seeding every lane with it would count it VL times for sum.
        result = EmitBind("reduce_update",
                          Call(plan.accumulator_dtype, VmiOp(fold_name),
                               {previous, result, one}),
                          &body);
      }
      body.push_back(Evaluate(Call(
          DataType::Void(), VmiOp("vstore"),
          {result, pointer(plan.destination, destination_indices(output), 1, 2),
           i32(0)})));
      Stmt output_loop = For(output, i32(0), i32(plan.output_extent),
                             ForKind::kSerial, SeqStmt::Flatten(body));
      Stmt compute = SeqStmt::Flatten(
          Bind(full_mask, mask(i32(std::min(plan.reduction_extent, lanes)))),
          output_loop);
      return SeqStmt::Flatten(AllocBuffer(accumulator),
                              AttrStmt(accumulator->data, "tl.vmi.alloc_kind",
                                       StringImm("data"), compute));
    }
    body.push_back(
        Bind(active_mask, mask(i32(plan.output_extent) - chunk * i32(lanes))));
    PrimExpr first = load(chunk * i32(lanes), i32(0), &body);
    first = seed(first, active_mask, &body);
    body.push_back(BufferStore(accumulator, first, {i32(0)}));
    Array<Stmt> update;
    PrimExpr next = load(chunk * i32(lanes), reduction, &update);
    PrimExpr folded = EmitBind("reduce_fold",
                               Call(vector_dtype, VmiOp(fold_name),
                                    {read_accumulator(), next, active_mask}),
                               &update);
    update.push_back(BufferStore(accumulator, folded, {i32(0)}));
    if (plan.reduction_extent > 1) {
      body.push_back(For(reduction, i32(1), i32(plan.reduction_extent - 1),
                         ForKind::kSerial, SeqStmt::Flatten(update)));
    }
    if (!plan.clear) {
      int64_t physical_lanes = 2048 / plan.accumulator_dtype.bits();
      int64_t footprint =
          ((lanes + physical_lanes - 1) / physical_lanes) * physical_lanes;
      PrimExpr previous = EmitBind(
          "reduce_previous",
          Call(vector_dtype, VmiOp("vload"),
               {pointer(plan.destination,
                        destination_indices(chunk * i32(lanes)), footprint, 1),
                i32(0)},
               {{"size", i32(lanes)}}),
          &body);
      PrimExpr combined =
          EmitBind("reduce_update",
                   Call(vector_dtype, VmiOp(fold_name),
                        {previous, read_accumulator(), active_mask}),
                   &body);
      body.push_back(BufferStore(accumulator, combined, {i32(0)}));
    }
    body.push_back(Evaluate(
        Call(DataType::Void(), VmiOp("vstore"),
             {read_accumulator(),
              pointer(plan.destination, destination_indices(chunk * i32(lanes)),
                      lanes, 2),
              i32(0), active_mask})));
    // Full chunks and the tail use the same vector reduction. Loads may read
    // complete physical lanes; active_mask limits the fold and output store.
    int64_t chunks = (plan.output_extent + lanes - 1) / lanes;
    Stmt compute = For(chunk, i32(0), i32(chunks), ForKind::kSerial,
                       SeqStmt::Flatten(body));
    return SeqStmt::Flatten(AllocBuffer(accumulator),
                            AttrStmt(accumulator->data, "tl.vmi.alloc_kind",
                                     StringImm("data"), compute));
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tl::attr::kBufferVersion) {
      BufferVersionMap saved = buffer_versions_;
      for (const auto &[storage, count] : op->node.cast<BufferVersionMap>()) {
        buffer_versions_.Set(storage, count);
      }
      Stmt result = StmtExprMutator::VisitStmt_(op);
      buffer_versions_ = std::move(saved);
      return result;
    }
    if (op->attr_key == tirx::attr::thread_extent) {
      if (const auto *iv = op->node.as<IterVarNode>()) {
        With<arith::ConstraintContext> launch_bounds(
            &analyzer_, iv->var >= 0 && iv->var < op->value);
        return StmtExprMutator::VisitStmt_(op);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AllocBufferNode *op) final {
    active_allocations_.push_back(op->buffer);
    return GetRef<Stmt>(op);
  }

  bool IsScalarFillParameter(const PrimExpr &value) const {
    if (value.dtype() != DataType::Float(16) &&
        value.dtype() != DataType::Float(32)) {
      return false;
    }
    if (const auto *parameter = value.as<VarNode>()) {
      return parameters_.count(GetRef<Var>(parameter)) != 0;
    }
    if (const auto *conversion = value.as<CastNode>()) {
      return IsScalarFillParameter(conversion->value);
    }
    return false;
  }

  Stmt LowerFill(const Call &call, int64_t lanes) {
    pto_analysis::PtoFillPlan plan =
        pto_analysis::PlanPtoFill(call, &analyzer_, buffer_versions_);
    auto i32 = [](int64_t value) { return IntImm(DataType::Int(32), value); };
    Buffer dst = plan.destination->buffer;
    DataType dtype = dst->dtype;
    Array<Stmt> statements;
    // Capture once at the source statement, before any destination store.
    // Constants stay literals so the VMI bridge retains their exact dtype.
    PrimExpr value = plan.value;
    bool scalar_parameter = IsScalarFillParameter(value);
    PrimExpr load_value = value;
    std::vector<DataType> float_casts;
    if (!scalar_parameter) {
      while (const auto *conversion = load_value.as<CastNode>()) {
        if (!(conversion->dtype.is_float() ||
              conversion->dtype.is_bfloat16()) ||
            !(conversion->value.dtype().is_float() ||
              conversion->value.dtype().is_bfloat16())) {
          break;
        }
        float_casts.push_back(conversion->dtype);
        load_value = conversion->value;
      }
    }
    if (!load_value.as<BufferLoadNode>() && !load_value.as<VarNode>()) {
      // Integer casts and constants retain their scalar lowering. Floating
      // UB snapshots and parameters use vector conversions below, preserving
      // every intermediate rounding rather than collapsing a cast chain.
      load_value = value;
      float_casts.clear();
    }
    if (const auto *load = load_value.as<BufferLoadNode>()) {
      // Capture the scalar once before any store; a later store may write
      // the same element from which this uniform value was loaded.
      PrimExpr address = Call(DataType::Handle(), tl::access_ptr(),
                              {GetRef<BufferLoad>(load), i32(1), i32(1)});
      value = EmitBind("fill_value",
                       Call(load_value.dtype(), VmiOp("vload"),
                            {address, i32(0)}, {{"size", i32(1)}}),
                       &statements);
    } else if (!scalar_parameter && !load_value.as<IntImmNode>() &&
               !load_value.as<FloatImmNode>()) {
      value = EmitBind("fill_value", load_value, &statements);
    }
    DataType broadcast_dtype = float_casts.empty() ? dtype : load_value.dtype();
    PrimExpr vector =
        EmitBind("fill_vector",
                 Call(broadcast_dtype.with_lanes(lanes), VmiOp("vbrc"), {value},
                      {{"size", i32(lanes)}}),
                 &statements);
    for (auto conversion = float_casts.rbegin();
         conversion != float_casts.rend(); ++conversion) {
      // Keep the scalar snapshot in its loaded dtype. PTOAS has no complete
      // layout route for narrow one-lane vcvt -> broadcast -> wide store;
      // broadcasting the uniform input before vcvt preserves the same cast
      // per element and uses the validated full-VL conversion route.
      vector =
          EmitBind("fill_cast",
                   Call(conversion->with_lanes(lanes), VmiOp("vcvt"), {vector},
                        {{"to_dtype", StringImm(TvmDtypeName(*conversion))}}),
                   &statements);
    }
    PrimExpr mask = EmitBind("fill_mask",
                             Call(DataType::Bool(lanes), VmiOp("create_mask"),
                                  {i32(lanes)}, {{"size", i32(lanes)}}),
                             &statements);
    Var row("fill_row", DataType::Int(32));
    Var chunk("fill_chunk", DataType::Int(32));
    Array<PrimExpr> indices;
    for (const Range &range : plan.destination->region) {
      indices.push_back(range->min);
    }
    if (plan.outer_axis >= 0) {
      indices.Set(plan.outer_axis, row);
    }
    indices.Set(plan.continuous_axis, chunk * i32(lanes));
    PrimExpr pointer = Call(DataType::Handle(), tl::access_ptr(),
                            {BufferLoad(dst, indices), i32(lanes), i32(2)});
    Stmt body = Evaluate(Call(DataType::Void(), VmiOp("vstore"),
                              {vector, pointer, i32(0), mask}));
    body = For(chunk, i32(0), i32(plan.run_extent / lanes), ForKind::kSerial,
               body);
    if (plan.outer_axis >= 0) {
      body = For(row, i32(0), i32(plan.outer_extent), ForKind::kSerial, body);
    }
    statements.push_back(body);
    return SeqStmt::Flatten(statements);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    analyzer_.Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (current_vf_lanes_ != 0) {
      if (const auto *call = op->value.as<CallNode>()) {
        if (pto_analysis::IsPtoDirectReduce(GetRef<Call>(call))) {
          return LowerDirectReduce(GetRef<Call>(call), current_vf_lanes_);
        }
        if (pto_analysis::IsPtoFill(GetRef<Call>(call))) {
          return LowerFill(GetRef<Call>(call), current_vf_lanes_);
        }
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  // Replace one statement (the unit's For) with its conversion, recursing
  // through the same wrapper kinds DiscoverUnits understands.
  Stmt ReplaceStmt(const Stmt &body, const For &target,
                   const pto_analysis::PtoInverseMapping &mapping) {
    // Conversion needs the owner environment at the unit's source position,
    // including allocations inside VF-local blocks and flat definitions.
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
                       body.as<AllocBufferNode>() == nullptr};
    if (const auto *alloc = body.as<AllocBufferNode>()) {
      active_allocations_.push_back(alloc->buffer);
      return body;
    }
    if (const auto *for_node = body.as<ForNode>()) {
      if (for_node == target.get()) {
        return ConvertUnit(target, mapping);
      }
      Stmt nb = ReplaceStmt(for_node->body, target, mapping);
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
        Stmt ns = ReplaceStmt(s, target, mapping);
        changed |= !ns.same_as(s);
        new_seq.push_back(ns);
      }
      return changed ? SeqStmt::Flatten(new_seq) : body;
    }
    if (const auto *attr = body.as<AttrStmtNode>()) {
      Stmt nb = ReplaceStmt(attr->body, target, mapping);
      if (nb.same_as(attr->body)) {
        return body;
      }
      return AttrStmt(attr->node, attr->attr_key, attr->value, nb);
    }
    if (const auto *block = body.as<SBlockNode>()) {
      for (const Buffer &buffer : block->alloc_buffers) {
        active_allocations_.push_back(buffer);
      }
      Stmt nb = ReplaceStmt(block->body, target, mapping);
      if (nb.same_as(block->body)) {
        return body;
      }
      return SBlock(block->iter_vars, block->reads, block->writes,
                    block->name_hint, nb, block->init, block->alloc_buffers,
                    block->match_buffers, block->annotations);
    }
    if (const auto *realize = body.as<SBlockRealizeNode>()) {
      SBlock block =
          Downcast<SBlock>(ReplaceStmt(realize->block, target, mapping));
      if (block.same_as(realize->block)) {
        return body;
      }
      return SBlockRealize(realize->iter_values, realize->predicate, block);
    }
    return body;
  }

  // ------------------------------------------------------------------
  // Per-unit conversion
  // ------------------------------------------------------------------
  Stmt ConvertUnit(const For &loop,
                   const pto_analysis::PtoInverseMapping &mapping) {
    // The inverse mapping comes from the region entry: it is the shared
    // source of truth for lanes, the mask plan and the body conversion
    // (see InvertPtoLaneLayout).
    bind_env_.clear();
    value_map_.clear();
    tmp_counter_ = 0;
    unit_mapping_ = mapping;
    Optional<For> inner;
    if (const auto *body_for = loop->body.as<ForNode>()) {
      if (body_for->kind == ForKind::kParallel) {
        inner = GetRef<For>(body_for);
      }
    }
    loop_var_ = mapping.is_2d ? (mapping.select_inner ? mapping.inner_var
                                                      : mapping.outer_var)
                              : loop->loop_var;
    lanes_ = mapping.lanes;
    lane_var_ = mapping.lane_var;
    chunk_analysis_ = mapping.chunk_var;
    lane_indexed_expr_ = mapping.index_expr;
    Var chunk("q", loop_var_.dtype());
    chunk_index_ = chunk;
    analyzer_.Bind(chunk, Range::FromMinExtent(
                              make_zero(chunk.dtype()),
                              IntImm(chunk.dtype(), mapping.chunk_count)));
    if (mapping.is_2d) {
      const Var kept =
          mapping.select_inner ? mapping.outer_var : mapping.inner_var;
      const int64_t extent =
          mapping.select_inner ? mapping.outer_extent : mapping.inner_extent;
      analyzer_.Bind(kept, Range::FromMinExtent(make_zero(kept.dtype()),
                                                IntImm(kept.dtype(), extent)));
    }

    // Decide once from the static extent. A tail-bearing unit keeps the
    // existing element-wise remaining-mask loop, including its full chunks.
    const bool has_tail = mapping.tail_lanes != 0;
    Buffer remaining;
    if (has_tail) {
      remaining =
          tirx::decl_buffer({IntImm(DataType::Int(32), 1)}, DataType::Int(32),
                            "remaining", "local.var");
      mask_var_ = Var("mask", DataType::Bool(static_cast<int>(lanes_)));
    } else {
      ICHECK(shared_mask_.defined());
      mask_var_ = shared_mask_;
    }

    Array<Stmt> statements;
    const Stmt &source = inner.defined() ? inner.value()->body : loop->body;
    ConvertBody(source, &statements);
    Stmt converted = DeadBindElim()(SeqStmt::Flatten(statements));
    Array<Stmt> chunk_body;
    if (has_tail) {
      PrimExpr count = BufferLoad(remaining, {make_zero(DataType::Int(32))});
      chunk_body.push_back(
          Bind(mask_var_, Call(mask_var_.dtype(), VmiOp("create_mask"), {count},
                               {{"size", IntImm(DataType::Int(32), lanes_)}})));
      chunk_body.push_back(converted);
      chunk_body.push_back(
          BufferStore(remaining, count - IntImm(DataType::Int(32), lanes_),
                      {make_zero(DataType::Int(32))}));
    } else {
      chunk_body.push_back(converted);
    }
    Stmt body = For(chunk, make_zero(chunk.dtype()),
                    IntImm(chunk.dtype(), mapping.chunk_count),
                    ForKind::kSerial, SeqStmt::Flatten(chunk_body));
    if (has_tail) {
      Stmt init = BufferStore(remaining,
                              IntImm(DataType::Int(32), mapping.logical_extent),
                              {make_zero(DataType::Int(32))});
      body = SeqStmt::Flatten(init, body);
    }
    if (inner.defined()) {
      // Reset remaining inside the kept loop so every row/column has the
      // same ceil(E/L) iterations and its own final partial mask.
      const Var kept_var =
          mapping.select_inner ? loop->loop_var : inner.value()->loop_var;
      const int64_t kept_extent =
          mapping.select_inner ? mapping.outer_extent : mapping.inner_extent;
      body = For(kept_var, make_zero(kept_var.dtype()),
                 IntImm(kept_var.dtype(), kept_extent), ForKind::kSerial, body);
    }
    return has_tail ? SeqStmt::Flatten(AllocBuffer(remaining), body) : body;
  }

  // Substitute the analysis chunk placeholder with the unified loop's q.
  PrimExpr BindChunkToQ(const PrimExpr &expr) const {
    return pto_analysis::SubstituteVar(expr, chunk_analysis_.get(),
                                       chunk_index_);
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

  PrimExpr Broadcast(PrimExpr value, Array<Stmt> *pre) {
    if (value.dtype().lanes() != 1)
      return value;
    return EmitBind("broadcast",
                    Call(VectorDType(value.dtype(), lanes_), VmiOp("vbrc"),
                         {value},
                         {{"size", IntImm(DataType::Int(32), lanes_)}}),
                    pre);
  }

  bool IsUniform(const PrimExpr &expr) {
    Optional<Var> kept_var;
    int64_t kept_extent = 0;
    if (unit_mapping_.is_2d) {
      kept_var = unit_mapping_.select_inner ? unit_mapping_.outer_var
                                            : unit_mapping_.inner_var;
      kept_extent = unit_mapping_.select_inner ? unit_mapping_.outer_extent
                                               : unit_mapping_.inner_extent;
    }
    return pto_analysis::IsPtoLaneUniform(
        expr, loop_var_, lane_indexed_expr_, lane_var_, chunk_analysis_, lanes_,
        unit_mapping_.chunk_count, kept_var, kept_extent, bind_env_);
  }

  PrimExpr ConvertScalarExpr(const PrimExpr &expr) {
    class ScalarRemap : public ExprMutator {
    public:
      explicit ScalarRemap(VectorizeParallelToPTOImpl *parent)
          : parent_(parent) {}
      PrimExpr VisitExpr_(const VarNode *op) final {
        auto it = parent_->value_map_.find(GetRef<Var>(op));
        if (it != parent_->value_map_.end()) {
          ICHECK_EQ((*it).second.dtype().lanes(), 1);
          return (*it).second;
        }
        return GetRef<PrimExpr>(op);
      }

    private:
      VectorizeParallelToPTOImpl *parent_;
    } remap(this);
    PrimExpr scalar = pto_analysis::SubstituteVar(
        expr, loop_var_.get(), BindChunkToQ(lane_indexed_expr_));
    scalar = analyzer_.Simplify(SubstituteLaneZero(scalar));
    return analyzer_.Simplify(remap(scalar));
  }

  PrimExpr ScalarBufferLoad(const Buffer &buffer,
                            const Array<PrimExpr> &indices) {
    bool remap_metadata = false;
    auto inspect = [&](const PrimExpr &expr) {
      PostOrderVisit(expr, [&](const ObjectRef &node) {
        if (const auto *var = node.as<VarNode>()) {
          remap_metadata |=
              var == loop_var_.get() || value_map_.count(GetRef<Var>(var));
        }
      });
    };
    for (const PrimExpr &shape : buffer->shape)
      inspect(shape);
    for (const PrimExpr &stride : buffer->strides)
      inspect(stride);
    inspect(buffer->elem_offset);
    if (remap_metadata) {
      // Flatten the complete address before remapping it: unused metadata
      // may mention a vector Bind even when the physical address is scalar.
      for (const Buffer &owner : active_allocations_) {
        if (owner->data.same_as(buffer->data) &&
            owner->dtype == buffer->dtype) {
          Buffer flat = owner.GetFlattenedBuffer();
          PrimExpr offset = ConvertScalarExpr(buffer->ElemOffset(indices)[0]);
          return BufferLoad(flat,
                            {analyzer_.Simplify(offset - flat->elem_offset)});
        }
      }
    }
    return BufferLoad(buffer, indices);
  }

  PrimExpr ConvertUint16Offset(const PrimExpr &expr, Array<Stmt> *pre) {
    const DataType dtype = DataType::UInt(16);
    const DataType vector_dtype = VectorDType(dtype, lanes_);
    if (const auto *integer = expr.as<IntImmNode>()) {
      return make_const(dtype, integer->value);
    }
    if (const auto *var = expr.as<VarNode>()) {
      if (var == loop_var_.get()) {
        PrimExpr base = analyzer_.Simplify(
            BindChunkToQ(SubstituteLaneZero(lane_indexed_expr_)));
        return EmitBind("offset_vci",
                        Call(vector_dtype, VmiOp("vci"), {cast(dtype, base)},
                             {{"size", IntImm(DataType::Int(32), lanes_)}}),
                        pre);
      }
      return cast(dtype, ConvertScalarExpr(expr));
    }
    const PrimExpr *lhs;
    const PrimExpr *rhs;
    std::string operation;
    if (const auto *add = expr.as<AddNode>()) {
      lhs = &add->a;
      rhs = &add->b;
      operation = "vadd";
    } else if (const auto *sub = expr.as<SubNode>()) {
      lhs = &sub->a;
      rhs = &sub->b;
      operation = "vsub";
    } else {
      const auto *mul = expr.as<MulNode>();
      ICHECK(mul) << "unsupported proved uint16 offset node "
                  << expr->GetTypeKey();
      lhs = &mul->a;
      rhs = &mul->b;
      operation = "vmul";
    }
    PrimExpr a = ConvertUint16Offset(*lhs, pre);
    PrimExpr b = ConvertUint16Offset(*rhs, pre);
    if (a.dtype().lanes() == 1 && b.dtype().lanes() == 1) {
      if (operation == "vadd") {
        return a + b;
      }
      if (operation == "vsub") {
        return a - b;
      }
      return a * b;
    }
    return EmitBind("offset",
                    Call(vector_dtype, VmiOp(operation),
                         {Broadcast(a, pre), Broadcast(b, pre), mask_var_}),
                    pre);
  }

  struct IndexedOperands {
    PrimExpr pointer;
    PrimExpr base;
    PrimExpr offsets;
  };

  IndexedOperands LowerIndexed(const pto_analysis::MemoryAccess &access,
                               const PrimExpr &mask, Array<Stmt> *pre) {
    pto_analysis::IndexedMemoryOwner owner =
        pto_analysis::GetIndexedMemoryOwner(access.buffer, active_allocations_,
                                            &analyzer_,
                                            "VectorizeParallelToPTO");
    pto_analysis::PtoIndexedAccessPlan plan =
        pto_analysis::PlanPtoIndexedAccess(access, owner, loop_var_,
                                           unit_mapping_, &analyzer_);
    PrimExpr base = plan.rebased ? ConvertScalarExpr(plan.base) : plan.base;
    PrimExpr offsets =
        Broadcast(plan.uint16_offsets ? ConvertUint16Offset(plan.offsets, pre)
                                      : ConvertExpr(plan.offsets, pre),
                  pre);
    // A tail-bearing unit sanitizes offsets in every iteration. Divisible
    // units pass the loaded/calculated offsets directly, without cleaning.
    if (unit_mapping_.tail_lanes != 0) {
      PrimExpr zero =
          EmitBind("offset_zero",
                   Call(offsets.dtype(), VmiOp("vbrc"),
                        {make_zero(offsets.dtype().element_of())},
                        {{"size", IntImm(DataType::Int(32), lanes_)}}),
                   pre);
      offsets = EmitBind(
          "safe_offsets",
          Call(offsets.dtype(), VmiOp("vsel"), {mask, offsets, zero}), pre);
    }
    // PTOAS normalizes signedness and widens narrow offset carriers.
    // PreparePtoIndexedOffset rejects implicit narrowing before conversion.
    return {pto_analysis::MakeIndexedMemoryDescriptor(owner, access.is_write),
            base, offsets};
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
      PrimExpr value = pto_analysis::IsPtoIndexDType(bind->value.dtype()) &&
                               IsUniform(bind->value)
                           ? ConvertScalarExpr(bind->value)
                           : ConvertExpr(bind->value, &pre);
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
      Array<Stmt> pre;
      PrimExpr value = ConvertExpr(store->value, &pre);
      auto access = pto_analysis::AnalyzeBufferAccess(
          store->buffer, store->indices, /*is_write=*/true, loop_var_,
          CurrentMapping(), &analyzer_, bind_env_);
      value = Broadcast(value, &pre);
      PrimExpr call;
      if (access.pattern == pto_analysis::AccessPattern::kContinuous) {
        Array<PrimExpr> starts;
        for (const PrimExpr &index : access.start_indices) {
          starts.push_back(ConvertScalarExpr(BindChunkToQ(index)));
        }
        PrimExpr ptr = Call(DataType::Handle(), tl::access_ptr(),
                            {ScalarBufferLoad(store->buffer, starts),
                             IntImm(DataType::Int(32), lanes_),
                             IntImm(DataType::Int(32), 2)});
        call = Call(DataType::Void(), VmiOp("vstore"),
                    {value, ptr, IntImm(DataType::Int(32), 0), mask_var_});
      } else {
        IndexedOperands operands = LowerIndexed(access, mask_var_, &pre);
        call = Call(DataType::Void(), VmiOp("vscatter"),
                    {value, operands.pointer, operands.base, operands.offsets,
                     mask_var_});
      }
      for (const Stmt &stmt : pre)
        out->push_back(stmt);
      out->push_back(Evaluate(call));
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
      auto access = pto_analysis::AnalyzeBufferAccess(
          load->buffer, load->indices, /*is_write=*/false, loop_var_,
          CurrentMapping(), &analyzer_, bind_env_);
      if (access.pattern == pto_analysis::AccessPattern::kContinuous) {
        Array<PrimExpr> start_idx;
        for (const auto &idx : access.start_indices) {
          start_idx.push_back(ConvertScalarExpr(BindChunkToQ(idx)));
        }
        PrimExpr ptr =
            Call(DataType::Handle(), tl::access_ptr(),
                 {ScalarBufferLoad(load->buffer, start_idx),
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
      if (access.pattern == pto_analysis::AccessPattern::kIndexed) {
        IndexedOperands operands = LowerIndexed(access, mask_var_, pre);
        return EmitBind(
            "gather",
            Call(VectorDType(load->dtype, lanes_), VmiOp("vgather"),
                 {operands.pointer, operands.base, operands.offsets, mask_var_},
                 {{"pmode", StringImm("zero")}}),
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
            ConvertScalarExpr(BindChunkToQ(SubstituteLaneZero(idx))));
      }
      return EmitBind("sload", ScalarBufferLoad(load->buffer, indices), pre);
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
    if (expr.as<ModNode>() || expr.as<MinNode>() || expr.as<MaxNode>() ||
        expr.as<SelectNode>()) {
      ICHECK(IsUniform(expr))
          << "[VectorizeParallelToPTO] internal error: unsupported "
             "lane-varying indexed coordinate passed Verify: "
          << expr->GetTypeKey();
      PrimExpr coordinate = pto_analysis::SubstituteVar(expr, loop_var_.get(),
                                                        lane_indexed_expr_);
      return ConvertScalarExpr(BindChunkToQ(coordinate));
    }
    if (const auto *call = expr.as<CallNode>()) {
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
    if (pto_analysis::IsPtoScalarCoordinate(expr, loop_var_, bind_env_)) {
      return ConvertScalarExpr(expr);
    }
    LOG(FATAL) << "[VectorizeParallelToPTO] internal error: unsupported "
                  "expression passed Verify: "
               << expr->GetTypeKey();
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
  PrimExpr chunk_index_; // unified chunk loop Var
  Var chunk_analysis_;   // analysis placeholder from the inverse mapping
  Var lane_var_;         // analysis placeholder; never escapes
  PrimExpr lane_indexed_expr_;
  /*! The complete inverse mapping of the unit being converted
   * (2D fields included), consumed by CurrentMapping(). */
  pto_analysis::PtoInverseMapping unit_mapping_;
  int64_t lanes_ = 0;
  Var mask_var_;
  int tmp_counter_ = 0;
  int64_t current_vf_lanes_ = 0;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> parameters_;
  std::vector<Buffer> active_allocations_;
  BufferVersionMap buffer_versions_;
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
