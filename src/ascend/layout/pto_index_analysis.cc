/*!
 * \file pto_index_analysis.cc
 * \brief Low-level read-only index/address analysis shared by PTO lane-layout
 * planning and the PTO Verify/Vectorize consumers (see pto_index_analysis.h).
 */

#include "pto_index_analysis.h"

#include <tvm/ir/op.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <functional>
#include <limits>
#include <sstream>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {
namespace pto {

using namespace tirx;
using ffi::GetRef;
using ffi::Map;
using ffi::ObjectRef;

namespace {

// Substitute `from` -> `to` inside an expression tree.
class VarSubstituter : public ExprMutator {
public:
  VarSubstituter(const VarNode *from, PrimExpr to)
      : from_(from), to_(std::move(to)) {}

private:
  PrimExpr VisitExpr_(const VarNode *op) final {
    if (op == from_) {
      return to_;
    }
    return ffi::GetRef<PrimExpr>(op);
  }
  const VarNode *from_;
  PrimExpr to_;
};

} // namespace

PrimExpr SubstituteVar(const PrimExpr &expr, const VarNode *from, PrimExpr to) {
  return VarSubstituter(from, std::move(to))(expr);
}

std::pair<int64_t, int64_t> CheckedPtoPadding(int64_t extent, int64_t lanes) {
  // The PTO lane loop indexes with int32, so any extent above the int32
  // range can never execute; rejecting it here (before the padding
  // addition) also makes an int64 overflow of extent + pad impossible:
  // with extent <= INT32_MAX and lanes <= 256, extent + pad stays far
  // below INT64_MAX.
  constexpr int64_t kMaxIndex = std::numeric_limits<int32_t>::max();
  if (extent > kMaxIndex) {
    std::ostringstream oss;
    oss << "PTO extent " << extent << " exceeds the int32 index space (max "
        << kMaxIndex << ")";
    throw PtoAnalysisError(oss.str());
  }
  const int64_t pad = extent % lanes == 0 ? 0 : lanes - extent % lanes;
  const int64_t padded = extent + pad;
  // Classified re-check: an extent within int32 still gets a distinct
  // diagnostic when the padded value itself crosses the boundary.
  if (padded > kMaxIndex) {
    std::ostringstream oss;
    oss << "PTO padded extent " << padded
        << " exceeds the int32 index space (max " << kMaxIndex << ")";
    throw PtoAnalysisError(oss.str());
  }
  return {pad, padded};
}

Optional<PrimExpr> ResolveIndexAliases(const PrimExpr &index,
                                       const Map<Var, PrimExpr> &bind_env) {
  if (bind_env.empty()) {
    return index;
  }
  // Load-dependent Bind definitions are values, including arithmetic aliases
  // of a saved load. Keep their identity instead of creating fresh UB reads.
  using VarSet =
      std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;
  Map<Var, Integer> load_dependency;
  VarSet inspecting;
  std::function<bool(const PrimExpr &, int)> depends_on_load =
      [&](const PrimExpr &expr, int depth) {
        if (depth > 64) {
          return false; // The ordinary alias budget diagnoses this chain.
        }
        bool result = false;
        PostOrderVisit(expr, [&](const ObjectRef &node) {
          if (node.as<BufferLoadNode>()) {
            result = true;
          } else if (const auto *vn = node.as<VarNode>()) {
            Var var = GetRef<Var>(vn);
            if (load_dependency.count(var)) {
              result |= load_dependency[var]->value != 0;
            } else if (bind_env.count(var) && !inspecting.count(var)) {
              inspecting.insert(var);
              bool dependency = depends_on_load(bind_env[var], depth + 1);
              load_dependency.Set(var, Integer(dependency ? 1 : 0));
              result |= dependency;
              inspecting.erase(var);
            }
          }
        });
        return result;
      };
  constexpr int kMaxDepth = 64;
  // Expression-size budget. The depth cap alone is not enough —
  // every substitution round can double the expression (x1 = i + i;
  // x2 = x1 + x1; ...), so a 26-deep chain already exploded to 2^26 nodes
  // and died minutes later on a raw tvm int32-literal assertion instead of
  // a diagnostic. Count the nodes after each round and reject once the
  // budget is exceeded. Legal chains (plain var aliases, (i + x) - x
  // style cancellations) stay far below the budget and resolve quickly.
  constexpr int64_t kMaxNodes = 4096;
  auto count_nodes = [](const PrimExpr &expr) {
    int64_t nodes = 0;
    PostOrderVisit(expr, [&](const ObjectRef &) { ++nodes; });
    return nodes;
  };
  bool exhausted = true;
  std::function<bool(PrimExpr &, int)> resolve = [&](PrimExpr &expr,
                                                     int depth) -> bool {
    if (depth > kMaxDepth) {
      exhausted = false;
      return false;
    }
    const VarNode *found = nullptr;
    PostOrderVisit(expr, [&](const ObjectRef &node) {
      if (found == nullptr) {
        if (auto *var_node = node.as<VarNode>()) {
          Var var = GetRef<Var>(var_node);
          if (bind_env.count(var) && !depends_on_load(bind_env[var], 0)) {
            found = var_node;
          }
        }
      }
    });
    if (found == nullptr) {
      return true;
    }
    const PrimExpr &def = bind_env[GetRef<Var>(found)];
    expr = SubstituteVar(expr, found, def);
    if (count_nodes(expr) > kMaxNodes) {
      std::ostringstream oss;
      oss << "bind alias resolution exceeded the expression-size budget ("
          << kMaxNodes
          << " nodes); the alias chain expands too much "
             "(e.g. doubling definitions like x = y + y)";
      throw PtoAnalysisError(oss.str());
    }
    return resolve(expr, depth + 1);
  };
  PrimExpr result = index;
  try {
    if (!resolve(result, 0) || !exhausted) {
      return Optional<PrimExpr>();
    }
  } catch (const ffi::Error &err) {
    // A tvm-internal failure during substitution (e.g. an int32
    // literal overflowing while an expression is rebuilt) must surface as
    // an analysis diagnostic, not a raw Check failed crash.
    throw PtoAnalysisError(
        std::string("bind alias resolution hit a tvm-internal error: ") +
        err.what());
  }
  return result;
}

bool IsOrdinaryUbScope(const std::string &scope) {
  return scope == "shared" || scope == "shared.dyn";
}

bool IsPtoIndexDType(DataType dtype) {
  return (dtype.is_int() || dtype.is_uint()) && dtype.lanes() == 1 &&
         (dtype.bits() == 16 || dtype.bits() == 32);
}

bool IsPtoScalarCoordinate(const PrimExpr &expr, const Var &vector_var,
                           const Map<Var, PrimExpr> &bind_env) {
  bool scalar = IsPtoIndexDType(expr.dtype());
  PostOrderVisit(expr, [&](const ObjectRef &node) {
    if (node.as<BufferLoadNode>() || node.as<CallNode>() ||
        node.as<CastNode>()) {
      scalar = false;
    } else if (const auto *op = node.as<VarNode>()) {
      Var var = GetRef<Var>(op);
      scalar &= var.dtype().lanes() == 1 && !var.same_as(vector_var) &&
                !bind_env.count(var);
    }
  });
  return scalar;
}

bool IsSupportedAddressExpr(const PrimExpr &expr) {
  // Existing coordinate arithmetic remains legal. Ordinary UB loads are
  // classified with their lane dependence separately; calls with no index
  // lowering and memory in other address spaces remain unsupported.
  class AddressCheck : public ExprVisitor {
  public:
    bool supported = true;
    void VisitExpr_(const CallNode *op) final {
      if (!supported) {
        return;
      }
      const auto *op_node = op->op.as<OpNode>();
      std::string name = op_node != nullptr ? op_node->name : "call";
      // Pure index intrinsics with defined semantics may appear; only
      // known-pure builtins are currently allowed.
      // call_extern/call_pure_extern/call_llvm_intrin etc. are rejected:
      // their effect is opaque to this pass.
      if (name == "tir.tvm_access_ptr" || name == "tl.access_ptr" ||
          name == "tir.address_of") {
        ExprVisitor::VisitExpr_(op);
        return;
      }
      supported = false;
    }
    void VisitExpr_(const BufferLoadNode *op) final {
      if (!IsOrdinaryUbScope(op->buffer.scope())) {
        supported = false;
        return;
      }
      ExprVisitor::VisitExpr_(op);
    }
    void VisitExprDefault_(const ffi::Object *) final { /* fine */ }
  };
  AddressCheck checker;
  checker(expr);
  return checker.supported;
}

namespace {

// This expression is used only for lane-dependence proofs. Replacing a loaded
// value by a symbolic scalar (uniform) or symbolic coefficient times lane
// (varying) avoids proving a varying saved load to be a uniform scalar base.
// All occurrences of one Bind reuse its symbol; separate source loads do not.
class LoadedValueProof : public ExprMutator {
public:
  LoadedValueProof(Var vector_var, PrimExpr inverse_index, Var lane_var,
                   const Map<Var, PrimExpr> &bind_env,
                   arith::Analyzer *analyzer)
      : vector_var_(std::move(vector_var)),
        inverse_index_(std::move(inverse_index)),
        lane_var_(std::move(lane_var)), bind_env_(bind_env),
        analyzer_(analyzer) {}

private:
  PrimExpr VisitExpr_(const VarNode *op) final {
    Var var = GetRef<Var>(op);
    if (var.same_as(vector_var_)) {
      return inverse_index_;
    }
    if (!bind_env_.count(var)) {
      return var;
    }
    if (symbols_.count(var)) {
      return symbols_[var];
    }
    if (visiting_.count(var) || visiting_.size() >= 64) {
      throw PtoAnalysisError("cyclic or too deep loaded index Bind definition");
    }
    visiting_.insert(var);
    PrimExpr def = VisitExpr(bind_env_[var]);
    visiting_.erase(var);
    // Preserve the entire definition's identity, even for an arithmetic alias
    // of a saved index; it is evaluated where the Bind appears in the source.
    PrimExpr result = MakeSymbol(var.dtype(), IsUniform(def));
    symbols_.Set(var, result);
    return result;
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    bool uniform = true;
    for (const PrimExpr &offset : op->buffer->ElemOffset(op->indices)) {
      uniform &= IsUniform(VisitExpr(offset));
    }
    return MakeSymbol(op->dtype, uniform);
  }

  bool IsUniform(const PrimExpr &expr) const {
    PrimExpr zero =
        SubstituteVar(expr, lane_var_.get(), make_zero(lane_var_.dtype()));
    return analyzer_->CanProveEqual(expr, zero);
  }

  PrimExpr MakeSymbol(DataType dtype, bool uniform) {
    if (!IsPtoIndexDType(dtype)) {
      throw PtoAnalysisError("loaded index values require scalar 16-bit or "
                             "32-bit integer elements");
    }
    Var symbol("pto_loaded_index", dtype);
    if (uniform) {
      return symbol;
    }
    // Give the intercept an independent symbol too: the saved value at lane
    // zero is not zero. Otherwise a product could incorrectly lose a varying
    // dependence just because the proof substituted lane=0.
    Var slope("pto_loaded_index_stride", dtype);
    return symbol + slope * Cast(dtype, lane_var_);
  }

  Var vector_var_;
  PrimExpr inverse_index_;
  Var lane_var_;
  const Map<Var, PrimExpr> &bind_env_;
  arith::Analyzer *analyzer_;
  Map<Var, PrimExpr> symbols_;
  std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> visiting_;
};

// Indexed addresses are emitted through the ordinary expression converter.
// Keep its integer arithmetic subset here instead of accepting every pure
// coordinate node merely because continuous scalar addressing can handle it.
class IndexedExprCheck : public ExprVisitor {
public:
  IndexedExprCheck(const Var &vector_var, const Map<Var, PrimExpr> &bind_env,
                   std::function<bool(const PrimExpr &)> is_uniform)
      : vector_var_(vector_var), bind_env_(bind_env),
        is_uniform_(std::move(is_uniform)) {}

  void Check(const PrimExpr &expr) {
    if (!IsPtoIndexDType(expr.dtype())) {
      throw PtoAnalysisError("indexed element offsets require scalar 16-bit "
                             "or 32-bit integer expressions");
    }
    if (IsPtoScalarCoordinate(expr, vector_var_, bind_env_)) {
      return;
    }
    bool scalar_coordinate =
        (expr.as<DivNode>() || expr.as<ModNode>() || expr.as<MinNode>() ||
         expr.as<MaxNode>() || expr.as<SelectNode>()) &&
        is_uniform_(expr);
    if (!(expr.as<IntImmNode>() || expr.as<VarNode>() ||
          expr.as<BufferLoadNode>() || expr.as<AddNode>() ||
          expr.as<SubNode>() || expr.as<MulNode>() || expr.as<CastNode>() ||
          scalar_coordinate)) {
      throw PtoAnalysisError(std::string("unsupported indexed offset node ") +
                             expr->GetTypeKey());
    }
    VisitExpr(expr);
  }

private:
  void VisitExpr_(const VarNode *op) final {
    Var var = GetRef<Var>(op);
    if (bind_env_.count(var)) {
      if (visiting_.count(var) || visiting_.size() >= 64) {
        throw PtoAnalysisError("cyclic or too deep loaded index definition");
      }
      visiting_.insert(var);
      Check(bind_env_[var]);
      visiting_.erase(var);
    }
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    if (!IsOrdinaryUbScope(op->buffer.scope())) {
      throw PtoAnalysisError(
          "gather/scatter indices must be loaded from "
          "ordinary UB buffers; Fragment and GM are unsupported");
    }
    if (!IsPtoIndexDType(op->dtype)) {
      throw PtoAnalysisError("gather/scatter indices must have 16-bit or "
                             "32-bit integer elements");
    }
    // This load's address is classified separately. Its coordinates are not
    // values emitted by ConvertExpr (for example, a buffer-version index).
  }

  void VisitExpr_(const CastNode *op) final { Check(op->value); }
  void VisitExpr_(const AddNode *op) final {
    Check(op->a);
    Check(op->b);
  }
  void VisitExpr_(const SubNode *op) final {
    Check(op->a);
    Check(op->b);
  }
  void VisitExpr_(const MulNode *op) final {
    Check(op->a);
    Check(op->b);
  }

  void VisitExpr_(const CallNode *) final {
    throw PtoAnalysisError("calls in indexed element offsets have no "
                           "supported PTO lowering");
  }

  void VisitExpr_(const DivNode *op) final {
    Check(op->a);
    Check(op->b);
  }
  void VisitExpr_(const ModNode *op) final {
    Check(op->a);
    Check(op->b);
  }
  void VisitExpr_(const MinNode *op) final {
    Check(op->a);
    Check(op->b);
  }
  void VisitExpr_(const MaxNode *op) final {
    Check(op->a);
    Check(op->b);
  }
  void VisitExpr_(const SelectNode *op) final {
    Check(op->true_value);
    Check(op->false_value);
    // The address whitelist has already excluded opaque condition calls.
  }

  const Var &vector_var_;
  const Map<Var, PrimExpr> &bind_env_;
  std::function<bool(const PrimExpr &)> is_uniform_;
  std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> visiting_;
};

} // namespace

bool IsPtoLaneUniform(const PrimExpr &expr, const Var &vector_var,
                      const PrimExpr &inverse_index, const Var &lane_var,
                      const Var &chunk_var, int64_t lanes, int64_t chunk_count,
                      const ffi::Optional<Var> &kept_var, int64_t kept_extent,
                      const Map<Var, PrimExpr> &bind_env) {
  arith::Analyzer proof;
  proof.Bind(lane_var, Range::FromMinExtent(make_zero(lane_var.dtype()),
                                            IntImm(lane_var.dtype(), lanes)));
  proof.Bind(chunk_var,
             Range::FromMinExtent(make_zero(chunk_var.dtype()),
                                  IntImm(chunk_var.dtype(), chunk_count)));
  if (kept_var.defined()) {
    proof.Bind(
        kept_var.value(),
        Range::FromMinExtent(make_zero(kept_var.value().dtype()),
                             IntImm(kept_var.value().dtype(), kept_extent)));
  }
  LoadedValueProof loaded(vector_var, inverse_index, lane_var, bind_env,
                          &proof);
  PrimExpr symbolic = loaded(expr);
  PrimExpr at_zero =
      SubstituteVar(symbolic, lane_var.get(), make_zero(lane_var.dtype()));
  return proof.CanProveEqual(symbolic, at_zero);
}

PtoAddressAnalysis
AnalyzePtoAddress(const Buffer &buffer, const ffi::Array<PrimExpr> &indices,
                  const Var &vector_var, const PrimExpr &inverse_index,
                  const Var &lane_var, const Var &chunk_var, int64_t lanes,
                  int64_t chunk_count, const ffi::Optional<Var> &kept_var,
                  int64_t kept_extent, const Map<Var, PrimExpr> &bind_env,
                  arith::Analyzer *analyzer) {
  PtoAddressAnalysis result;
  ffi::Array<PrimExpr> resolved;
  for (const PrimExpr &index : indices) {
    Optional<PrimExpr> alias = ResolveIndexAliases(index, bind_env);
    if (!alias.defined()) {
      throw PtoAnalysisError("index Bind alias chain is too deep or cyclic");
    }
    if (!IsSupportedAddressExpr(alias.value())) {
      throw PtoAnalysisError("address expression contains an unsupported "
                             "call or a non-UB index load");
    }
    resolved.push_back(alias.value());
    PrimExpr inverse =
        SubstituteVar(alias.value(), vector_var.get(), inverse_index);
    result.indices_after_inverse.push_back(inverse);
    result.start_indices.push_back(
        SubstituteVar(inverse, lane_var.get(), make_zero(lane_var.dtype())));
  }
  auto offset = [&](const ffi::Array<PrimExpr> &at) {
    ffi::Array<PrimExpr> offsets = buffer->ElemOffset(at);
    if (offsets.size() != 1) {
      throw PtoAnalysisError(
          "ElemOffset must return exactly one physical axis");
    }
    return offsets[0];
  };
  result.source_indices = resolved;
  result.original_offset = offset(resolved);

  arith::Analyzer proof;
  proof.Bind(lane_var, Range::FromMinExtent(make_zero(lane_var.dtype()),
                                            IntImm(lane_var.dtype(), lanes)));
  proof.Bind(chunk_var,
             Range::FromMinExtent(make_zero(chunk_var.dtype()),
                                  IntImm(chunk_var.dtype(), chunk_count)));
  if (kept_var.defined()) {
    proof.Bind(
        kept_var.value(),
        Range::FromMinExtent(make_zero(kept_var.value().dtype()),
                             IntImm(kept_var.value().dtype(), kept_extent)));
  }
  LoadedValueProof loaded(vector_var, inverse_index, lane_var, bind_env,
                          &proof);
  ffi::Array<PrimExpr> base_indices;
  std::vector<bool> varying_axes;
  for (size_t axis = 0; axis < resolved.size(); ++axis) {
    PrimExpr symbolic_index = loaded(resolved[axis]);
    PrimExpr at_zero = SubstituteVar(symbolic_index, lane_var.get(),
                                     make_zero(lane_var.dtype()));
    const bool uniform = proof.CanProveEqual(symbolic_index, at_zero);
    base_indices.push_back(uniform ? resolved[axis]
                                   : make_zero(resolved[axis].dtype()));
    varying_axes.push_back(!uniform);
  }
  result.scalar_base_offset = analyzer->Simplify(offset(base_indices));
  // Buffer metadata participates in the physical address too. Uniform
  // logical indices do not make a lane-dependent stride/origin a scalar base.
  PrimExpr symbolic_base = loaded(result.scalar_base_offset);
  PrimExpr base_at_zero =
      SubstituteVar(symbolic_base, lane_var.get(), make_zero(lane_var.dtype()));
  const bool scalar_base_uniform =
      proof.CanProveEqual(symbolic_base, base_at_zero);
  // This is a representability proof, not source address arithmetic. Do not
  // subtract two ElemOffsets: their source int32 operations may wrap a large
  // physical span into a seemingly valid ui16 window. Widen each operand and
  // bound every intermediate before multiplying/adding. A span above the ui16
  // limit (or one we cannot prove) only needs an out-of-range sentinel.
  constexpr int kMaxWindow = 65536;
  PrimExpr window = make_const(DataType::Int(64), 1);
  PrimExpr stride = make_const(DataType::Int(64), 1);
  for (size_t axis = resolved.size(); axis-- > 0;) {
    PrimExpr axis_stride =
        buffer->strides.empty() ? stride : buffer->strides[axis];
    PrimExpr count = Cast(DataType::Int(64), buffer->shape[axis]) - 1;
    if (varying_axes[axis] && !analyzer->CanProveEqual(count, 0)) {
      PrimExpr pitch = Cast(DataType::Int(64), axis_stride);
      if (analyzer->CanProveEqual(pitch, 0)) {
        continue;
      }
      if (!analyzer->CanProve(count >= 0) ||
          !analyzer->CanProve(count < kMaxWindow) ||
          !analyzer->CanProve(pitch >= 0) ||
          !analyzer->CanProve(pitch < kMaxWindow)) {
        window = make_const(DataType::Int(64), kMaxWindow + 1);
        break;
      }
      // Both factors are now less than 65536, so the int64 product and sum
      // cannot overflow, even when the source Buffer metadata is int32.
      window = analyzer->Simplify(window + count * pitch);
      if (!analyzer->CanProve(window <= kMaxWindow)) {
        window = make_const(DataType::Int(64), kMaxWindow + 1);
        break;
      }
    }
    if (buffer->strides.empty()) {
      PrimExpr shape = Cast(DataType::Int(64), buffer->shape[axis]);
      // A larger implicit pitch cannot contribute to a ui16 window on a
      // non-singleton varying axis. Cap it before forming a large product.
      if (analyzer->CanProve(shape > 0) &&
          analyzer->CanProve(shape <= kMaxWindow) &&
          analyzer->CanProve(stride <= kMaxWindow)) {
        stride = analyzer->Simplify(stride * shape);
      } else {
        stride = make_const(DataType::Int(64), kMaxWindow);
      }
    }
  }
  if (!scalar_base_uniform) {
    result.scalar_base_offset = make_zero(result.scalar_base_offset.dtype());
    window = make_const(DataType::Int(64), kMaxWindow + 1);
  }
  result.relative_extent = window;
  PrimExpr symbolic = loaded(result.original_offset);
  PrimExpr start =
      SubstituteVar(symbolic, lane_var.get(), make_zero(lane_var.dtype()));
  if (proof.CanProveEqual(symbolic, start + lane_var)) {
    result.pattern = PtoAddressPattern::kContinuous;
  } else if (proof.CanProveEqual(symbolic, start)) {
    result.pattern = PtoAddressPattern::kLaneUniform;
  } else {
    const DataType dtype = buffer->dtype;
    if (dtype.lanes() != 1 || (dtype.bits() != 16 && dtype.bits() != 32) ||
        !(dtype.is_int() || dtype.is_uint() || dtype.is_float() ||
          dtype.is_bfloat16())) {
      throw PtoAnalysisError(
          "indexed gather/scatter requires B16 or B32 payload");
    }
    auto is_uniform = [&](const PrimExpr &expr) {
      PrimExpr value = loaded(expr);
      PrimExpr zero =
          SubstituteVar(value, lane_var.get(), make_zero(lane_var.dtype()));
      return proof.CanProveEqual(value, zero);
    };
    IndexedExprCheck checker(vector_var, bind_env, is_uniform);
    checker.Check(result.original_offset);
    result.pattern = PtoAddressPattern::kIndexed;
  }
  return result;
}

} // namespace pto
} // namespace tl
} // namespace tvm
