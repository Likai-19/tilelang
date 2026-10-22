/*!
 * \file pto_fragment_layout_solver.cc
 * \brief Parity-based joint layout solver for PTO fragment buffers.
 * See pto_fragment_layout_solver.h for the model; the design contract is
 * the R4 sub-design (v3, §2–§3). Every rejection below is a classified
 * user-facing error; parity conflicts are *proven* conflicts (an odd
 * cycle over binary variables), never search failures.
 */

#include "pto_fragment_layout_solver.h"

#include <tvm/arith/analyzer.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../op/pto_index_analysis.h"
#include "parallel_to_pto_utils.h"

namespace tvm {
namespace tl {
namespace pto {

namespace {

using tirx::Buffer;
using tirx::BufferLoadNode;
using tirx::BufferStoreNode;
using tirx::For;
using tirx::ForKind;
using tirx::ForNode;
using tirx::SBlockNode;
using tirx::Stmt;
using tirx::StmtExprVisitor;

[[noreturn]] void Reject(const std::string &message) {
  std::ostringstream oss;
  oss << "[PtoFragmentLayout] " << message;
  LOG(FATAL) << oss.str();
}

bool IsFragmentScope(const Buffer &buffer) {
  return buffer.scope() == "local.fragment";
}

bool IsUbScope(const Buffer &buffer) {
  const std::string scope = buffer.scope();
  return scope == "shared" || scope == "shared.dyn";
}

int64_t ConstantExtent(const PrimExpr &extent, const std::string &what) {
  const int64_t *v = as_const_int(extent);
  if (v == nullptr || *v <= 0) {
    std::ostringstream oss;
    oss << what << " extent must be a compile-time positive constant, got "
        << extent;
    Reject(oss.str());
  }
  return *v;
}

// ---------------------------------------------------------------------------
// Access scan (mirrors the consumers' lexical rules: flat-sequence binds,
// AttrStmt scoping)
// ---------------------------------------------------------------------------

struct UnitAccess {
  Buffer buffer;
  Array<PrimExpr> indices;
  bool is_write;
  Map<Var, PrimExpr> bind_env;
};

class AccessScanner : public StmtExprVisitor {
public:
  std::vector<UnitAccess> accesses;

  void VisitStmt_(const BufferStoreNode *op) final {
    accesses.push_back({op->buffer, op->indices, /*is_write=*/true, bind_env_});
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const BufferLoadNode *op) final {
    accesses.push_back(
        {op->buffer, op->indices, /*is_write=*/false, bind_env_});
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const tirx::BindNode *op) final {
    StmtExprVisitor::VisitStmt_(op);
    bind_env_.Set(op->var, op->value);
  }
  void VisitStmt_(const tirx::AttrStmtNode *op) final {
    Map<Var, PrimExpr> saved = bind_env_;
    StmtExprVisitor::VisitStmt_(op);
    bind_env_ = std::move(saved);
  }

private:
  Map<Var, PrimExpr> bind_env_;
};

// ---------------------------------------------------------------------------
// Region / unit model
// ---------------------------------------------------------------------------

struct UnitInfo {
  For loop;                 // outermost Parallel For of the chain
  const ForNode *inner;     // inner Parallel For (2D only)
  bool is_2d = false;
  Var iv, jv;               // outer / inner loop vars (1D: iv == jv)
  int64_t M = 1, N = 1;     // outer / inner extents (1D: N == 1)
  int ordinal = 0;          // source order within the region
  std::vector<UnitAccess> accesses;  // innermost-body accesses
  std::string ub_why[2];    // per-direction UB failure reason (2D only)
};

struct RegionInfo {
  const SBlockNode *block = nullptr;
  int64_t lanes = 0;
  std::vector<UnitInfo> units;
  std::vector<Buffer> fragments;  // distinct: accessed by units OR allocated
};

const ForNode *InnerParallelOf(const For &unit) {
  const auto *body_for = unit->body.as<ForNode>();
  if (body_for != nullptr && body_for->kind == ForKind::kParallel) {
    return body_for;
  }
  return nullptr;
}

// ---------------------------------------------------------------------------
// UB direction feasibility: the continuity proof only holds for shared
// scopes (a fragment's logical row-major stride says nothing about its
// backing layout — that is the point of this solver).
// ---------------------------------------------------------------------------

void ProbeDirection(const UnitInfo &unit, bool select_inner, int64_t lanes,
                    bool *ok, std::string *why) {
  const Var &sel_var = select_inner ? unit.jv : unit.iv;
  const Var &kept_var = select_inner ? unit.iv : unit.jv;
  const int64_t sel_extent = select_inner ? unit.N : unit.M;
  const int64_t kept_extent = select_inner ? unit.M : unit.N;
  const auto [pad, padded_extent] = CheckedPtoPadding(sel_extent, lanes);
  (void)pad;
  const int64_t chunk_count = padded_extent / lanes;
  DataType dt = sel_var.dtype();
  Var lane("solver_lane", dt);
  Var q("solver_q", dt);
  PrimExpr sel_expr = q * IntImm(dt, static_cast<int>(lanes)) + lane;
  *ok = true;
  for (const auto &access : unit.accesses) {
    if (!IsUbScope(access.buffer) || IsLaneScalarBuffer(access.buffer)) {
      continue;
    }
    Array<PrimExpr> resolved;
    for (const auto &idx : access.indices) {
      Optional<PrimExpr> opt;
      try {
        opt = ResolveIndexAliases(idx, access.bind_env);
      } catch (const PtoAnalysisError &err) {
        *ok = false;
        *why = err.what();
        return;
      }
      if (!opt.has_value()) {
        *ok = false;
        *why = "bind alias chain in index is too deep, cyclic, or depends "
               "on a loaded value";
        return;
      }
      resolved.push_back(opt.value());
    }
    for (const auto &idx : resolved) {
      if (!IsSupportedAddressExpr(idx)) {
        *ok = false;
        *why = "address expression contains a loaded value or an "
               "unsupported call";
        return;
      }
    }
    Array<PrimExpr> indices_after;
    for (const auto &idx : resolved) {
      indices_after.push_back(SubstituteVar(idx, sel_var.get(), sel_expr));
    }
    Array<PrimExpr> raw = access.buffer->ElemOffset(indices_after);
    if (raw.size() != 1) {
      *ok = false;
      *why = "multi-axis element offsets are not supported";
      return;
    }
    PrimExpr off = raw[0];
    Array<PrimExpr> start_indices;
    for (const auto &idx : indices_after) {
      start_indices.push_back(SubstituteVar(idx, lane.get(), make_zero(dt)));
    }
    PrimExpr start = access.buffer->ElemOffset(start_indices)[0];
    arith::Analyzer probe;
    probe.Bind(lane, Range::FromMinExtent(make_zero(dt), IntImm(dt, lanes)));
    probe.Bind(
        q, Range::FromMinExtent(make_zero(dt), IntImm(dt, chunk_count)));
    probe.Bind(kept_var,
               Range::FromMinExtent(make_zero(dt), IntImm(dt, kept_extent)));
    const bool continuous = probe.CanProveEqual(off, start + lane);
    const bool uniform = probe.CanProveEqual(
        off, probe.Simplify(SubstituteVar(off, lane.get(), make_zero(dt))));
    if (!continuous && !(uniform && !access.is_write)) {
      std::ostringstream oss;
      oss << "buffer `" << access.buffer->name << "` ("
          << (access.is_write ? "store" : "load")
          << ") is neither continuous nor lane-uniform under the "
          << (select_inner ? "inner (j)" : "outer (i)") << " mapping";
      *ok = false;
      *why = oss.str();
      return;
    }
  }
}

// Feasible direction mask for the UB accesses of one 2D unit:
// bit0 = inner (j vectorized), bit1 = outer (i vectorized).
int UbFeasibleDirections(UnitInfo *unit, int64_t lanes) {
  int mask = 0;
  for (int candidate = 0; candidate < 2; ++candidate) {
    bool ok = false;
    std::string why;
    ProbeDirection(*unit, /*select_inner=*/candidate == 0, lanes, &ok, &why);
    if (ok) {
      mask |= (1 << candidate);
    } else {
      unit->ub_why[candidate] = why;
    }
  }
  return mask;
}

// ---------------------------------------------------------------------------
// Parity union-find over binary variables
// ---------------------------------------------------------------------------

class ParityDSU {
public:
  explicit ParityDSU(size_t n) : parent_(n), parity_(n, 0) {
    for (size_t i = 0; i < n; ++i) {
      parent_[i] = static_cast<int>(i);
    }
  }

  // Returns the root; *parity_to_root gets value(v) XOR value(root).
  int Find(int v, int *parity_to_root) {
    int parity = 0;
    int cur = v;
    while (parent_[cur] != cur) {
      parity ^= parity_[cur];
      cur = parent_[cur];
    }
    // Path compression (second pass, parity-preserving).
    int node = v;
    int acc = parity;
    while (parent_[node] != node) {
      int next = parent_[node];
      int old_p = parity_[node];
      parent_[node] = cur;
      parity_[node] = acc;
      acc ^= old_p;
      node = next;
    }
    *parity_to_root = parity;
    return cur;
  }

  // Unite with value(a) XOR value(b) == relation. Returns false on
  // contradiction with an existing edge.
  bool Unite(int a, int b, int relation) {
    int pa = 0, pb = 0;
    int ra = Find(a, &pa);
    int rb = Find(b, &pb);
    if (ra == rb) {
      return (pa ^ pb) == relation;
    }
    parent_[ra] = rb;
    parity_[ra] = relation ^ pa ^ pb;
    return true;
  }

private:
  std::vector<int> parent_;
  std::vector<int> parity_;  // XOR to parent
};

// Canonical layout construction — the same expression form
// InferPtoSimdLayout builds, so injected and derived proposals agree.
Fragment BuildCanonicalLayout(const Buffer &buffer, int lane_dim,
                              int64_t lanes, const Range &lane_bounds,
                              int64_t *entry_count_out) {
  std::vector<int64_t> dims;
  for (const auto &s : buffer->shape) {
    dims.push_back(ConstantExtent(
        s, "fragment buffer `" + buffer->name + "`"));
  }
  if (dims.size() > 2u) {
    std::ostringstream oss;
    oss << "fragment buffer `" << buffer->name << "` of rank "
        << dims.size()
        << " is not supported yet; PTO slot storage covers 1D/2D "
           "fragments only";
    Reject(oss.str());
  }
  const int64_t sel = dims[lane_dim];
  const auto [pad, padded_sel] = CheckedPtoPadding(sel, lanes);
  (void)pad;
  const int64_t per_dim_entries = padded_sel / lanes;
  int64_t entry_count = per_dim_entries;
  for (size_t d = 0; d < dims.size(); ++d) {
    if (static_cast<int>(d) != lane_dim) {
      entry_count *= dims[d];
    }
  }
  if (entry_count > std::numeric_limits<int32_t>::max()) {
    std::ostringstream oss;
    oss << "fragment buffer `" << buffer->name << "` needs " << entry_count
        << " slot entries, overflowing the int32 index space";
    Reject(oss.str());
  }
  const DataType ity = DataType::Int(32);
  PrimExpr lane_e = FloorMod(InputPlaceholder(static_cast<size_t>(lane_dim)),
                             IntImm(ity, static_cast<int>(lanes)));
  PrimExpr entry_e = FloorDiv(InputPlaceholder(static_cast<size_t>(lane_dim)),
                              IntImm(ity, static_cast<int>(lanes)));
  if (dims.size() == 2u) {
    const int other_dim = lane_dim == 0 ? 1 : 0;
    entry_e = InputPlaceholder(static_cast<size_t>(other_dim)) *
                  IntImm(ity, static_cast<int>(per_dim_entries)) +
              entry_e;
  }
  if (entry_count_out != nullptr) {
    *entry_count_out = entry_count;
  }
  return Fragment(buffer->shape, {entry_e}, lane_e, Integer(1), std::nullopt)
      ->BindThreadRange(lane_bounds);
}

int64_t BackingBytes(int64_t entry_count, int64_t lanes,
                     const DataType &dtype) {
  return entry_count * lanes * ((dtype.bits() + 7) / 8);
}

// ---------------------------------------------------------------------------
// The solver
// ---------------------------------------------------------------------------

class FragmentLayoutSolver {
public:
  explicit FragmentLayoutSolver(const LayoutMap &annotated)
      : annotated_(annotated) {}

  std::optional<LayoutMap> Run(const Stmt &body, bool target_is_pto) {
    if (!target_is_pto) {
      return std::nullopt;
    }
    CollectRegions(body);
    if (regions_.empty()) {
      return std::nullopt;
    }
    PreScan(body);
    BuildConstraints();
    Solve();
    auto out = EmitLayouts();
    return out;
  }

private:
  // ------ collection ------
  void CollectRegions(const Stmt &body) {
    class RegionCollector : public tirx::StmtVisitor {
    public:
      explicit RegionCollector(FragmentLayoutSolver *outer) : outer_(outer) {}
      void VisitStmt_(const SBlockNode *op) final {
        if (op->name_hint == kSimdVfBlockName) {
          PtoRegionScan scan = ScanPtoRegion(op->body);
          if (scan.has_any_parallel) {
            outer_->AddRegion(op, scan);
          }
          return;  // Nested regions are handled at their own visit.
        }
        tirx::StmtVisitor::VisitStmt_(op);
      }

    private:
      FragmentLayoutSolver *outer_;
    };
    RegionCollector collector(this);
    collector(body);
  }

  void AddRegion(const SBlockNode *block, const PtoRegionScan &scan) {
    RegionInfo info;
    info.block = block;
    auto lanes_anno = block->annotations.Get("tl.simdvf_lanes");
    if (!lanes_anno.has_value()) {
      // LayoutInference owns the definitive diagnostic for a converting
      // region without lanes; the solver skips malformed input.
      return;
    }
    auto lanes_imm = lanes_anno.value().as<IntImmNode>();
    if (lanes_imm == nullptr || lanes_imm->value <= 0) {
      return;
    }
    info.lanes = lanes_imm->value;

    std::set<const tirx::BufferNode *> seen_fragments;
    for (const For &unit : scan.units) {
      UnitInfo ui;
      ui.loop = unit;
      ui.inner = InnerParallelOf(unit);
      ui.is_2d = ui.inner != nullptr;
      ui.iv = unit->loop_var;
      ui.M = ConstantExtent(unit->extent, "parallel loop");
      if (ui.is_2d) {
        ui.jv = ui.inner->loop_var;
        ui.N = ConstantExtent(ui.inner->extent, "inner parallel loop");
      } else {
        ui.jv = ui.iv;
        ui.N = 1;
      }
      ui.ordinal = static_cast<int>(info.units.size());
      const Stmt &unit_body =
          ui.is_2d ? static_cast<const Stmt &>(ui.inner->body)
                   : static_cast<const Stmt &>(unit->body);
      AccessScanner scanner;
      scanner(unit_body);
      ui.accesses = std::move(scanner.accesses);
      for (const auto &access : ui.accesses) {
        if (IsFragmentScope(access.buffer) &&
            seen_fragments.insert(access.buffer.get()).second) {
          info.fragments.push_back(access.buffer);
        }
      }
      unit_owners_[unit.get()] = info.units.size();
      info.units.push_back(std::move(ui));
    }
    // Allocated-but-unaccessed fragments (floating, B5): they still need
    // a layout before the engine's step 0 can seed them.
    for (const Buffer &buffer : block->alloc_buffers) {
      if (IsFragmentScope(buffer) &&
          seen_fragments.insert(buffer.get()).second) {
        info.fragments.push_back(buffer);
      }
    }
    regions_.push_back(std::move(info));
  }

  // ------ pre-scan ------
  void PreScan(const Stmt &body) {
    class PreScanner : public StmtExprVisitor {
    public:
      explicit PreScanner(FragmentLayoutSolver *outer) : outer_(outer) {}

      void VisitStmt_(const SBlockNode *op) final {
        const bool entering_region = outer_->region_index_of_.count(op) != 0;
        if (entering_region) {
          // SimdVF-scope layout annotations targeting region fragments
          // are rejected outright (the engine silently ignores them).
          for (const auto &kv : op->annotations) {
            if (kv.first != std::string(attr::kLayoutMap)) {
              continue;
            }
            auto map = kv.second.as<Map<Var, Layout>>();
            if (!map.has_value()) {
              continue;
            }
            for (const auto &[var, layout] : map.value()) {
              if (outer_->IsRegionFragmentData(var.get())) {
                Reject("a layout annotation attached to the SIMD_VF region "
                       "itself is not supported; annotate at kernel scope "
                       "or let the solver derive the layout");
              }
            }
          }
          ++region_depth_;
        }
        StmtExprVisitor::VisitStmt_(op);
        if (entering_region) {
          --region_depth_;
        }
      }

      void VisitStmt_(const ForNode *op) final {
        // Entering a converting unit's own loop: everything under it
        // down to the unit body is a legal fragment-access site.
        const bool is_unit = region_depth_ > 0 &&
                             outer_->unit_owners_.count(op) != 0;
        if (is_unit) {
          ++unit_depth_;
        }
        StmtExprVisitor::VisitStmt_(op);
        if (is_unit) {
          --unit_depth_;
        }
      }

      void VisitStmt_(const BufferStoreNode *op) final {
        CheckAccess(op->buffer);
        if (op->predicate.defined()) {
          PostOrderVisit(op->predicate, [&](const ObjectRef &n) {
            if (const auto *load = n.as<BufferLoadNode>()) {
              CheckAccess(load->buffer);
            }
          });
        }
        StmtExprVisitor::VisitStmt_(op);
      }

      void VisitExpr_(const BufferLoadNode *op) final {
        CheckAccess(op->buffer);
        StmtExprVisitor::VisitExpr_(op);
      }

      void VisitExpr_(const tirx::CallNode *op) final {
        const auto *op_node = op->op.as<OpNode>();
        if (op_node != nullptr &&
            std::string(op_node->name).rfind("tl.tileop.", 0) == 0) {
          bool touches_fragment = false;
          PostOrderVisit(GetRef<tirx::Call>(op), [&](const ObjectRef &n) {
            if (const auto *load = n.as<BufferLoadNode>()) {
              if (IsFragmentScope(load->buffer)) {
                touches_fragment = true;
              }
            }
          });
          if (touches_fragment && outer_->HasRegionFragmentIn(
                                       GetRef<tirx::Call>(op))) {
            Reject("a tile operator touching a fragment buffer survived "
                   "(fragment fill/copy are expanded before layout "
                   "inference); fragment tile operators need a PTO "
                   "contract to participate");
          }
        }
        StmtExprVisitor::VisitExpr_(op);
      }

    private:
      void CheckAccess(const Buffer &buffer) {
        if (!IsFragmentScope(buffer) || !outer_->IsRegionFragment(buffer)) {
          return;
        }
        if (unit_depth_ == 0) {
          std::ostringstream oss;
          oss << "fragment buffer `" << buffer->name
              << "` is accessed outside a converting T.Parallel body; "
                 "every fragment access of a PTO kernel must live inside "
                 "a T.Parallel unit of a SIMD_VF region";
          Reject(oss.str());
        }
      }

      FragmentLayoutSolver *outer_;
      int region_depth_ = 0;
      int unit_depth_ = 0;
    };

    // Region membership index for the scanner above.
    for (size_t r = 0; r < regions_.size(); ++r) {
      region_index_of_[regions_[r].block] = static_cast<int>(r) + 1;
    }
    PreScanner scanner(this);
    scanner(body);

    CheckSameDataSharing();
    CheckCrossRegionSharing();
    LoadPinnedAnnotations();
  }

  void CheckSameDataSharing() {
    std::unordered_map<const VarNode *, const tirx::BufferNode *> by_data;
    for (const RegionInfo &region : regions_) {
      for (const Buffer &buffer : region.fragments) {
        const VarNode *data = buffer->data.get();
        auto it = by_data.find(data);
        if (it == by_data.end()) {
          by_data.emplace(data, buffer.get());
        } else if (it->second != buffer.get()) {
          std::ostringstream oss;
          oss << "fragment storage view sharing is not supported: `"
              << it->second->name << "` and `" << buffer->name
              << "` are views of one allocation";
          Reject(oss.str());
        }
      }
    }
  }

  void CheckCrossRegionSharing() {
    std::unordered_map<const tirx::BufferNode *, size_t> owner;
    for (size_t r = 0; r < regions_.size(); ++r) {
      for (const Buffer &buffer : regions_[r].fragments) {
        auto [it, inserted] = owner.emplace(buffer.get(), r);
        if (!inserted && it->second != r) {
          std::ostringstream oss;
          oss << "fragment buffer `" << buffer->name
              << "` is used by more than one SIMD_VF region; PTO slot "
                 "storage requires each fragment to stay inside a single "
                 "region";
          Reject(oss.str());
        }
      }
    }
  }

  void LoadPinnedAnnotations() {
    for (const auto &[buffer, layout] : annotated_) {
      if (!IsRegionFragment(buffer)) {
        continue;
      }
      auto frag_opt = layout.as<Fragment>();
      if (!frag_opt.has_value()) {
        std::ostringstream oss;
        oss << "annotated layout for fragment buffer `" << buffer->name
            << "` is not a Fragment; PTO slot storage requires the "
               "canonical fragment form";
        Reject(oss.str());
      }
      const Fragment frag = frag_opt.value();
      const int lane_dim = PtoFragmentLaneDim(frag);
      const int64_t lanes = LanesOf(buffer);
      if (lane_dim < 0) {
        std::ostringstream oss;
        oss << "annotated layout for fragment buffer `" << buffer->name
            << "` is not the canonical PTO form (the storage lane must be "
               "a floor-mod of exactly one input dimension); drop the "
               "annotation or annotate the canonical form";
        Reject(oss.str());
      }
      int64_t count = 0;
      Fragment canonical = BuildCanonicalLayout(
          buffer, lane_dim, lanes, LaneBounds(lanes), &count);
      if (!canonical->IsEqual(frag.get())) {
        std::ostringstream oss;
        oss << "annotated layout for fragment buffer `" << buffer->name
            << "` disagrees with the canonical form for lane dim "
            << lane_dim
            << " (replicate / thread range / entry shape); only the "
               "canonical PTO annotation form is accepted";
        Reject(oss.str());
      }
      pinned_lane_dim_[buffer.get()] = lane_dim;
    }
  }

  // ------ constraints & solve ------
  void BuildConstraints() {
    // Variable indexing: one binary variable per 2D unit and per 2D
    // fragment; 1D units/fragments are constants.
    // var_unit_ / var_buffer_ are indexed by the var id, so every var
    // creation pushes into both (a placeholder in the unused one).
    for (const RegionInfo &region : regions_) {
      for (const UnitInfo &unit : region.units) {
        if (unit.is_2d) {
          unit_var_[unit.loop.get()] = static_cast<int>(vars_.size());
          vars_.push_back(VarKind::kUnit);
          var_unit_.push_back(&unit);
          var_buffer_.push_back(Buffer());
        }
      }
      for (const Buffer &buffer : region.fragments) {
        if (buffer->shape.size() == 2u) {
          frag_var_[buffer.get()] = static_cast<int>(vars_.size());
          vars_.push_back(VarKind::kFragment);
          var_unit_.push_back(nullptr);
          var_buffer_.push_back(buffer);
          frag_lanes_[buffer.get()] = region.lanes;
        }
      }
    }
    dsu_ = std::make_unique<ParityDSU>(vars_.size());

    for (RegionInfo &region : regions_) {
      for (UnitInfo &unit : region.units) {
        ConstrainUnit(unit);
      }
    }
  }

  void ConstrainUnit(UnitInfo &unit) {
    int mask = unit.is_2d ? 0b11 : 0b01;
    if (unit.is_2d) {
      mask = UbFeasibleDirections(&unit, LanesOfRegionOf(unit));
      if (mask == 0) {
        std::ostringstream oss;
        oss << "proven layout conflict: the 2D T.Parallel unit ("
            << unit.iv->name_hint << ", " << unit.jv->name_hint
            << ") has no feasible vectorization direction for its "
               "shared-buffer accesses — inner: "
            << unit.ub_why[0] << "; outer: " << unit.ub_why[1];
        Reject(oss.str());
      }
    }

    // Fragment access constraints: under a feasible direction, exactly
    // one buffer dim may depend on the vectorized coordinate, and all
    // accesses of that buffer in this unit must agree on the dim.
    std::map<const tirx::BufferNode *, std::pair<int, int>> frag_lane;
    for (const auto &access : unit.accesses) {
      if (!IsFragmentScope(access.buffer)) {
        continue;
      }
      for (int candidate = 0; candidate < 2; ++candidate) {
        if ((mask & (1 << candidate)) == 0) {
          continue;
        }
        const bool select_inner = candidate == 0;
        const Var &sel_var = select_inner ? unit.jv : unit.iv;
        int dep_dim = -1;
        int deps = 0;
        for (size_t d = 0; d < access.indices.size(); ++d) {
          if (tirx::UsesVar(access.indices[d], [&](const VarNode *v) {
                return v == sel_var.get();
              })) {
            ++deps;
            dep_dim = static_cast<int>(d);
          }
        }
        auto inserted = frag_lane.emplace(
            access.buffer.get(), std::make_pair(-1, -1));
        int &recorded =
            candidate == 0 ? inserted.first->second.first
                           : inserted.first->second.second;
        if (deps == 0 || deps > 1 ||
            (recorded != -1 && recorded != dep_dim)) {
          if (unit.is_2d) {
            mask &= ~(1 << candidate);
            recorded = -1;  // direction dropped; value irrelevant
            break;
          }
          std::ostringstream oss;
          oss << "proven layout conflict: fragment buffer `"
              << access.buffer->name << "` is accessed with indices "
              << access.indices
              << " that cannot be served by whole-entry addressing under "
                 "the vectorized coordinate `"
              << sel_var->name_hint
              << "` (a 1D unit cannot change direction)";
          Reject(oss.str());
        }
        recorded = dep_dim;
      }
    }

    if (unit.is_2d) {
      if (mask == 0) {
        std::ostringstream oss;
        oss << "proven layout conflict: the 2D T.Parallel unit ("
            << unit.iv->name_hint << ", " << unit.jv->name_hint
            << ") has no feasible direction once the fragment access "
               "constraints are combined with its shared-buffer accesses";
        Reject(oss.str());
      }
      if (mask == 0b01 || mask == 0b10) {
        AddUnary(unit_var_.at(unit.loop.get()), mask == 0b01 ? 0 : 1);
      }
      for (const auto &[buf_node, lanes_pair] : frag_lane) {
        const int l_inner = lanes_pair.first;
        const int l_outer = lanes_pair.second;
        const bool inner_ok = (mask & 1) != 0 && l_inner >= 0;
        const bool outer_ok = (mask & 2) != 0 && l_outer >= 0;
        auto fit = frag_var_.find(buf_node);
        const bool frag_2d = fit != frag_var_.end();
        if (!inner_ok && !outer_ok) {
          continue;
        }
        if (inner_ok && outer_ok) {
          if (!frag_2d || l_inner == l_outer) {
            if (frag_2d) {
              AddUnary(fit->second, l_inner);
            }
            continue;
          }
          const int uvar = unit_var_.at(unit.loop.get());
          if (!dsu_->Unite(uvar, fit->second, /*relation=*/l_inner)) {
            std::ostringstream oss;
            oss << "proven layout conflict: the accesses of T.Parallel "
                   "unit ("
                << unit.iv->name_hint << ", " << unit.jv->name_hint
                << ") force fragment `" << buf_node->name
                << "` onto mutually flipped lane dimensions (odd parity "
                   "cycle)";
            Reject(oss.str());
          }
        } else if (inner_ok) {
          AddUnary(unit_var_.at(unit.loop.get()), 0);
          if (frag_2d) {
            AddUnary(fit->second, l_inner);
          }
        } else {
          AddUnary(unit_var_.at(unit.loop.get()), 1);
          if (frag_2d) {
            AddUnary(fit->second, l_outer);
          }
        }
      }
    } else {
      for (const auto &[buf_node, lanes_pair] : frag_lane) {
        auto fit = frag_var_.find(buf_node);
        if (fit != frag_var_.end() && lanes_pair.first >= 0) {
          AddUnary(fit->second, lanes_pair.first);
        }
      }
    }
  }

  void AddUnary(int var, int value) {
    int parity = 0;
    int root = dsu_->Find(var, &parity);
    auto it = root_value_.find(root);
    if (it == root_value_.end()) {
      root_value_[root] = value ^ parity;
      return;
    }
    if (it->second != (value ^ parity)) {
      std::ostringstream oss;
      oss << "proven layout conflict: contradictory lane-direction "
             "constraints reach the same unit/buffer group (the two "
             "candidate solutions both violate a pinned or "
             "direction-forced constraint)";
      Reject(oss.str());
    }
  }

  void Solve() {
    // Pinned annotations first.
    for (const auto &[buf_node, lane_dim] : pinned_lane_dim_) {
      auto it = frag_var_.find(buf_node);
      if (it != frag_var_.end()) {
        AddUnary(it->second, lane_dim);
      }
    }
    // Components without a value: pick by total backing bytes; ties by
    // the source-order-first unit with a free direction taking the
    // inner dimension. All units of a component share one region
    // (cross-region fragments were rejected in the pre-scan), so the
    // ordinal comparison is meaningful.
    for (int v = 0; v < static_cast<int>(vars_.size()); ++v) {
      int parity = 0;
      int root = dsu_->Find(v, &parity);
      if (root_value_.count(root) != 0) {
        continue;
      }
      int64_t bytes[2] = {0, 0};
      const UnitInfo *first_unit = nullptr;
      for (int w = 0; w < static_cast<int>(vars_.size()); ++w) {
        int w_parity = 0;
        int w_root = dsu_->Find(w, &w_parity);
        if (w_root != root) {
          continue;
        }
        if (vars_[w] == VarKind::kFragment) {
          const Buffer &buffer = var_buffer_[w];
          const int64_t lanes = frag_lanes_.at(buffer.get());
          for (int root_val = 0; root_val < 2; ++root_val) {
            int64_t count = 0;
            BuildCanonicalLayout(buffer, root_val ^ w_parity, lanes,
                                 LaneBounds(lanes), &count);
            bytes[root_val] += BackingBytes(count, lanes, buffer->dtype);
          }
        } else {
          const UnitInfo *candidate = var_unit_[w];
          if (first_unit == nullptr ||
              candidate->ordinal < first_unit->ordinal) {
            first_unit = candidate;
          }
        }
      }
      int chosen;
      if (bytes[0] < bytes[1]) {
        chosen = 0;
      } else if (bytes[1] < bytes[0]) {
        chosen = 1;
      } else if (first_unit != nullptr) {
        int u_parity = 0;
        dsu_->Find(unit_var_.at(first_unit->loop.get()), &u_parity);
        chosen = u_parity;  // unit value 0 (inner) -> root value
      } else {
        chosen = 0;
      }
      root_value_[root] = chosen;
    }
  }

  std::optional<LayoutMap> EmitLayouts() {
    LayoutMap out;
    for (const RegionInfo &region : regions_) {
      const Range lane_bounds = LaneBounds(region.lanes);
      for (const Buffer &buffer : region.fragments) {
        int lane_dim = 0;
        auto pit = pinned_lane_dim_.find(buffer.get());
        if (pit != pinned_lane_dim_.end()) {
          lane_dim = pit->second;
        } else if (buffer->shape.size() == 2u) {
          auto it = frag_var_.find(buffer.get());
          if (it != frag_var_.end()) {
            int parity = 0;
            int root = dsu_->Find(it->second, &parity);
            lane_dim = root_value_.at(root) ^ parity;
          } else {
            // Unconstrained 2D fragment (floating): byte-minimal lane
            // dim, ties on the last dim (the inner preference).
            const int64_t d0 =
                ConstantExtent(buffer->shape[0], "fragment buffer");
            const int64_t d1 =
                ConstantExtent(buffer->shape[1], "fragment buffer");
            const int64_t count0 =
                (d0 + region.lanes - 1) / region.lanes * d1;
            const int64_t count1 =
                (d1 + region.lanes - 1) / region.lanes * d0;
            lane_dim = count1 <= count0 ? 1 : 0;
          }
        }
        int64_t count = 0;
        Fragment layout = BuildCanonicalLayout(buffer, lane_dim,
                                               region.lanes, lane_bounds,
                                               &count);
        out.Set(buffer, layout);
      }
    }
    return out;
  }

  // ------ helpers ------
  bool IsRegionFragment(const Buffer &buffer) const {
    for (const RegionInfo &region : regions_) {
      for (const Buffer &frag : region.fragments) {
        if (frag.get() == buffer.get()) {
          return true;
        }
      }
    }
    return false;
  }

  bool HasRegionFragmentIn(const tirx::Call &call) const {
    bool found = false;
    PostOrderVisit(call, [&](const ObjectRef &n) {
      if (const auto *load = n.as<BufferLoadNode>()) {
        if (IsRegionFragment(load->buffer)) {
          found = true;
        }
      }
    });
    return found;
  }

  bool IsRegionFragmentData(const VarNode *data) const {
    for (const RegionInfo &region : regions_) {
      for (const Buffer &frag : region.fragments) {
        if (frag->data.get() == data) {
          return true;
        }
      }
    }
    return false;
  }

  int64_t LanesOf(const Buffer &buffer) const {
    for (const RegionInfo &region : regions_) {
      for (const Buffer &frag : region.fragments) {
        if (frag.get() == buffer.get()) {
          return region.lanes;
        }
      }
    }
    return 128;  // Only reached for region fragments.
  }

  int64_t LanesOfRegionOf(const UnitInfo &unit) const {
    for (const RegionInfo &region : regions_) {
      for (const UnitInfo &u : region.units) {
        if (u.loop.get() == unit.loop.get()) {
          return region.lanes;
        }
      }
    }
    return 128;
  }

  Range LaneBounds(int64_t lanes) const {
    return Range::FromMinExtent(make_zero(DataType::Int(32)),
                                IntImm(DataType::Int(32), lanes));
  }

  enum class VarKind { kUnit, kFragment };
  std::vector<VarKind> vars_;
  std::vector<const UnitInfo *> var_unit_;
  std::vector<Buffer> var_buffer_;
  std::unordered_map<const ForNode *, int> unit_var_;
  std::unordered_map<const tirx::BufferNode *, int> frag_var_;
  std::unordered_map<const tirx::BufferNode *, int> pinned_lane_dim_;
  std::unordered_map<const tirx::BufferNode *, int64_t> frag_lanes_;
  std::unordered_map<const ForNode *, size_t> unit_owners_;
  std::unordered_map<const SBlockNode *, int> region_index_of_;
  std::unique_ptr<ParityDSU> dsu_;
  std::unordered_map<int, int> root_value_;
  std::vector<RegionInfo> regions_;
  const LayoutMap &annotated_;
};

}  // namespace

// The lane dim of a canonical layout: the input dim whose placeholder
// the forward_thread expression depends on. InputPlaceholder vars are
// cached singletons, so pointer identity is stable.
int PtoFragmentLaneDim(const Fragment &layout) {
  int lane_dim = -1;
  for (int d = 0; d < layout->InputDim(); ++d) {
    const VarNode *ph = InputPlaceholder(static_cast<size_t>(d)).get();
    bool uses = false;
    PostOrderVisit(layout->GetForwardThread(), [&](const ObjectRef &n) {
      if (const auto *v = n.as<VarNode>()) {
        if (v == ph) {
          uses = true;
        }
      }
    });
    if (uses) {
      if (lane_dim != -1) {
        return -1;  // not the canonical single-dim form
      }
      lane_dim = d;
    }
  }
  return lane_dim;
}

std::optional<LayoutMap> SolvePtoFragmentLayouts(const Stmt &body,
                                                 bool target_is_pto,
                                                 const LayoutMap &annotated) {
  FragmentLayoutSolver solver(annotated);
  return solver.Run(body, target_is_pto);
}

}  // namespace pto
}  // namespace tl
}  // namespace tvm
