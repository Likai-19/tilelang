/*!
 * \file parallel_to_pto_utils.cc
 * \brief Shared read-only analysis for VerifyParallelToPTO and
 * VectorizeParallelToPTO.
 */

#include "parallel_to_pto_utils.h"

#include <tvm/arith/pattern.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include "../../layout/utils.h"
#include "../../op/utils.h"

#include <functional>
#include <memory>
#include <sstream>

namespace tvm {
namespace tl {
namespace pto {

using namespace tirx;
using ffi::Array;
using ffi::GetRef;
using ffi::Map;

bool IsPtoDirectReduce(const Call &call) {
  return call->op.same_as(Op::Get("tl.tileop.reduce"));
}

bool IsPtoFill(const Call &call) {
  return call->op.same_as(Op::Get("tl.tileop.fill"));
}

namespace {

// These helpers decode buffer metadata without validating TileOp admission or
// constructing a lowering plan. Verify and Vectorize interpret the same
// compiler-generated version prefix and physical strides.
int PtoVersionPrefix(const BufferRegion &region, arith::Analyzer *analyzer,
                     const Map<Var, int> &buffer_versions) {
  const Buffer &buffer = region->buffer;
  auto versions = buffer_versions.Get(buffer->data);
  if (versions.has_value() && versions.value() > 1 &&
      buffer->shape.size() > 1 &&
      analyzer->CanProveEqual(buffer->shape[0], versions.value()) &&
      analyzer->CanProveEqual(region->region[0]->extent, 1)) {
    return 1;
  }
  return 0;
}

Array<PrimExpr> PtoBufferStrides(const Buffer &buffer,
                                 arith::Analyzer *analyzer) {
  if (!buffer->strides.empty()) {
    return buffer->strides;
  }
  Array<PrimExpr> result;
  PrimExpr stride = Integer(1);
  for (size_t i = buffer->shape.size(); i-- > 0;) {
    result.insert(result.begin(), stride);
    stride = analyzer->Simplify(stride * buffer->shape[i]);
  }
  return result;
}

int PtoFillContinuousAxis(const BufferRegion &destination,
                          const Array<PrimExpr> &strides, int prefix,
                          arith::Analyzer *analyzer) {
  // A trailing singleton can have stride 1 as well. Prefer an axis with
  // actual elements so a compact column such as [128, 1] remains one full run.
  for (int axis = static_cast<int>(strides.size()) - 1; axis >= prefix;
       --axis) {
    if (analyzer->CanProveEqual(strides[axis], 1) &&
        !analyzer->CanProveEqual(destination->region[axis]->extent, 1)) {
      return axis;
    }
  }
  int axis = static_cast<int>(strides.size()) - 1;
  return analyzer->CanProveEqual(strides[axis], 1) ? axis : prefix;
}

} // namespace

void CheckPtoFill(const Call &call, int64_t lanes, arith::Analyzer *analyzer,
                  const Map<Var, int> &buffer_versions) {
  auto fail = [](const std::string &message) {
    throw PtoAnalysisError("fill/clear: " + message);
  };
  if (call->args.size() != 2) {
    fail("expected destination region and scalar value");
  }
  BufferRegion destination = NormalizeToBufferRegion(call->args[0]);
  Buffer dst = destination->buffer;
  if ((dst.scope() != "shared" && dst.scope() != "shared.dyn") ||
      !dst->axis_separators.empty()) {
    fail("only ordinary UB (shared/shared.dyn) destinations are supported; "
         "Fragment fill/clear is not supported");
  }
  DataType dtype = dst->dtype;
  if (dtype.lanes() != 1 ||
      !((dtype.is_float() && (dtype.bits() == 16 || dtype.bits() == 32)) ||
        dtype.is_bfloat16() ||
        ((dtype.is_int() || dtype.is_uint()) &&
         (dtype.bits() == 8 || dtype.bits() == 16 || dtype.bits() == 32)))) {
    fail("unsupported destination dtype");
  }
  int prefix = PtoVersionPrefix(destination, analyzer, buffer_versions);
  int rank = static_cast<int>(dst->shape.size()) - prefix;
  if (rank != 1 && rank != 2) {
    fail("requires a logical 1D/2D destination");
  }
  auto positive_constant = [&](const PrimExpr &expr) {
    const auto *value = expr.as<IntImmNode>();
    if (!value || value->value <= 0) {
      fail("buffer shapes and strides must be positive compile-time integers");
    }
    return value->value;
  };
  for (size_t i = prefix; i < dst->shape.size(); ++i) {
    positive_constant(dst->shape[i]);
    const Range &range = destination->region[i];
    if (!analyzer->CanProveEqual(range->min, 0)) {
      fail("nonzero destination region starts are not supported; use a "
           "zero-origin region");
    }
    positive_constant(range->extent);
    if (!analyzer->CanProve(range->extent <= dst->shape[i])) {
      fail("destination region must be provably in bounds");
    }
  }
  Array<PrimExpr> strides = PtoBufferStrides(dst, analyzer);
  for (const PrimExpr &stride : strides) {
    positive_constant(stride);
  }
  if (!analyzer->CanProveEqual(dst->elem_offset, 0)) {
    fail("nonzero destination storage offsets are not supported; use a "
         "zero-origin destination view");
  }
  int64_t alignment = 256 / dtype.bits();
  if (prefix != 0 &&
      !analyzer->CanProveEqual(floormod(strides[0], alignment), 0)) {
    fail("destination version slots must be 32-byte aligned");
  }
  int axis = PtoFillContinuousAxis(destination, strides, prefix, analyzer);
  if (!analyzer->CanProveEqual(strides[axis], 1)) {
    fail("requires a unit-stride axis");
  }
  int64_t run_extent =
      Downcast<IntImm>(destination->region[axis]->extent)->value;
  if (run_extent % lanes != 0) {
    fail("tail regions are not supported; the contiguous extent must be a "
         "multiple of SimdVF lanes");
  }
  if (rank == 2) {
    int outer_axis = prefix + (axis == prefix ? 1 : 0);
    if (!analyzer->CanProveEqual(destination->region[outer_axis]->extent, 1)) {
      if (positive_constant(strides[outer_axis]) <
          positive_constant(dst->shape[axis])) {
        fail("destination strides must describe non-overlapping rows or "
             "columns");
      }
      if (!analyzer->CanProveEqual(floormod(strides[outer_axis], alignment),
                                   0)) {
        fail("destination row or column starts must be 32-byte aligned");
      }
    }
  }
  PrimExpr value = analyzer->Simplify(cast(dtype, call->args[1]));
  std::function<void(const PrimExpr &)> check_value = [&](const PrimExpr
                                                              &value) {
    if (!value.dtype().is_scalar()) {
      fail("fill value must be a uniform scalar");
    }
    if (value.as<IntImmNode>() || value.as<FloatImmNode>() ||
        value.as<VarNode>()) {
      return;
    }
    if (const auto *load = value.as<BufferLoadNode>()) {
      if (load->buffer.scope() != "shared" &&
          load->buffer.scope() != "shared.dyn") {
        fail("scalar fill-value reads must use ordinary UB");
      }
      if (load->predicate.defined()) {
        fail("predicated scalar fill-value reads are not supported");
      }
      for (size_t i = 0; i < load->indices.size(); ++i) {
        if (SideEffect(load->indices[i]) != CallEffectKind::kPure ||
            !analyzer->CanProve(load->indices[i] >= 0) ||
            !analyzer->CanProve(load->indices[i] < load->buffer->shape[i])) {
          fail("scalar fill-value indices must be pure and provably in bounds");
        }
      }
      return;
    }
    if (const auto *conversion = value.as<CastNode>()) {
      CastContext context;
      context.value_kind = CastValueKind::kScalar;
      context.logical_lanes = 1;
      std::string reason;
      if (!PlanValueCast(conversion, context, &reason).has_value()) {
        fail(reason);
      }
      check_value(conversion->value);
      return;
    }
    fail("fill value must be a constant, uniform scalar or one UB read with "
         "supported scalar casts");
  };
  check_value(value);
}

PtoFillPlan PlanPtoFill(const Call &call, arith::Analyzer *analyzer,
                        const Map<Var, int> &buffer_versions) {
  // Verify owns admission and diagnostics. Planning consumes its established
  // invariants without rerunning the checks or changing the input IR.
  PtoFillPlan plan;
  plan.destination = NormalizeToBufferRegion(call->args[0]);
  const Buffer &dst = plan.destination->buffer;
  int prefix = PtoVersionPrefix(plan.destination, analyzer, buffer_versions);
  int rank = static_cast<int>(dst->shape.size()) - prefix;
  Array<PrimExpr> strides = PtoBufferStrides(dst, analyzer);
  plan.continuous_axis =
      PtoFillContinuousAxis(plan.destination, strides, prefix, analyzer);
  plan.run_extent =
      Downcast<IntImm>(plan.destination->region[plan.continuous_axis]->extent)
          ->value;
  if (rank == 2) {
    plan.outer_axis = prefix + (plan.continuous_axis == prefix ? 1 : 0);
    plan.outer_extent =
        Downcast<IntImm>(plan.destination->region[plan.outer_axis]->extent)
            ->value;
  }
  plan.value = analyzer->Simplify(cast(dst->dtype, call->args[1]));
  return plan;
}

void CheckPtoDirectReduce(const Call &call, arith::Analyzer *analyzer,
                          const Map<Var, int> &buffer_versions) {
  auto fail = [&](const std::string &message) {
    throw PtoAnalysisError("direct reduce: " + message);
  };
  if (call->args.size() != 5) {
    fail("expected source, destination, operation, dim and clear");
  }
  const auto *kind = call->args[2].as<StringImmNode>();
  const auto *dim = call->args[3].as<IntImmNode>();
  const auto *clear = call->args[4].as<IntImmNode>();
  if (!kind || !dim || !clear || (clear->value != 0 && clear->value != 1)) {
    fail("requires constant dim, operation and boolean clear");
  }
  std::string operation = kind->value;
  if (operation != "sum" && operation != "max" && operation != "min" &&
      operation != "abssum" && operation != "absmax") {
    fail("only sum, max, min, abssum and absmax are supported");
  }
  for (const auto &[key, value] : call->annotations) {
    if (key == "batch" || key == "nan_propagate") {
      const auto *integer = value.as<IntImmNode>();
      if (!integer || integer->value != (key == "batch" ? 1 : 0)) {
        fail("requires batch=1 and nan_propagate=False");
      }
    }
  }
  BufferRegion source = NormalizeToBufferRegion(call->args[0]);
  BufferRegion destination = NormalizeToBufferRegion(call->args[1]);
  Buffer src = source->buffer;
  Buffer dst = destination->buffer;
  auto ub = [](const Buffer &buffer) {
    return buffer.scope() == "shared" || buffer.scope() == "shared.dyn";
  };
  if (!ub(src) || !ub(dst) || !src->axis_separators.empty() ||
      !dst->axis_separators.empty()) {
    fail("source and destination must be ordinary linear UB buffers");
  }
  if (src->data.same_as(dst->data)) {
    fail("overlapping source/destination storage is not supported");
  }
  DataType from = src->dtype;
  DataType to = dst->dtype;
  bool same =
      from == to && ((to.is_float() && (to.bits() == 16 || to.bits() == 32)) ||
                     ((to.is_int() || to.is_uint()) && to.bits() == 32));
  bool widen = to == DataType::Float(32) &&
               (from == DataType::Float(16) || from.is_bfloat16());
  if ((!same && !widen) || from.lanes() != 1 || to.lanes() != 1) {
    fail("unsupported source/accumulator dtype combination");
  }
  auto logical_shape = [&](const BufferRegion &region, int *prefix) {
    const Buffer &buffer = region->buffer;
    if (region->region.size() != buffer->shape.size()) {
      fail("region rank does not match its buffer");
    }
    *prefix = PtoVersionPrefix(region, analyzer, buffer_versions);
    Array<PrimExpr> shape;
    for (size_t i = *prefix; i < buffer->shape.size(); ++i) {
      const Range &range = region->region[i];
      if (!analyzer->CanProveEqual(range->min, 0) ||
          !analyzer->CanProveEqual(range->extent, buffer->shape[i])) {
        fail("requires complete logical source and destination regions");
      }
      shape.push_back(buffer->shape[i]);
    }
    return shape;
  };
  int source_prefix = 0;
  int destination_prefix = 0;
  Array<PrimExpr> src_shape = logical_shape(source, &source_prefix);
  Array<PrimExpr> dst_shape = logical_shape(destination, &destination_prefix);
  int rank = src_shape.size();
  if ((rank != 1 && rank != 2) || dim->value < 0 || dim->value >= rank) {
    fail("requires a logical 1D/2D source and a valid reduction dim");
  }
  auto positive = [&](const PrimExpr &expr) {
    const auto *integer = expr.as<IntImmNode>();
    if (!integer || integer->value <= 0) {
      fail("all logical extents must be positive compile-time integers");
    }
    return integer->value;
  };
  for (const PrimExpr &extent : src_shape) {
    positive(extent);
  }
  for (const PrimExpr &extent : dst_shape) {
    positive(extent);
  }
  int reduction_axis = dim->value;
  int output_axis = rank == 1 ? 0 : 1 - reduction_axis;
  int64_t output_extent = rank == 1 ? 1 : positive(src_shape[output_axis]);
  bool keep_dims = dst_shape.size() == src_shape.size();
  if (rank == 1) {
    if (dst_shape.size() != 1 || positive(dst_shape[0]) != 1) {
      fail("1D reduction requires destination shape [1]");
    }
  } else if (keep_dims) {
    if (positive(dst_shape[reduction_axis]) != 1 ||
        positive(dst_shape[output_axis]) != output_extent) {
      fail("keepdims destination shape does not match the reduction");
    }
  } else if (dst_shape.size() != 1 || positive(dst_shape[0]) != output_extent) {
    fail("destination shape does not match the reduction");
  }
  int destination_output_axis =
      destination_prefix + (keep_dims && rank == 2 ? output_axis : 0);
  Array<PrimExpr> src_strides = PtoBufferStrides(src, analyzer);
  Array<PrimExpr> dst_strides = PtoBufferStrides(dst, analyzer);
  for (const PrimExpr &stride : src_strides) {
    positive(stride);
  }
  for (const PrimExpr &stride : dst_strides) {
    positive(stride);
  }
  if (rank == 2) {
    int64_t rows = positive(src_shape[0]);
    int64_t columns = positive(src_shape[1]);
    int64_t row_stride = positive(src_strides[source_prefix]);
    int64_t column_stride = positive(src_strides[source_prefix + 1]);
    if (!((column_stride == 1 && row_stride >= columns) ||
          (row_stride == 1 && column_stride >= rows))) {
      fail("source strides must describe non-overlapping rows or columns");
    }
  }
  bool reduce_across_lanes =
      analyzer->CanProveEqual(src_strides[source_prefix + reduction_axis], 1);
  if (!reduce_across_lanes &&
      (rank != 2 ||
       !analyzer->CanProveEqual(src_strides[source_prefix + output_axis], 1) ||
       !analyzer->CanProveEqual(dst_strides[destination_output_axis], 1))) {
    fail("no supported unit-stride reduction/output axis");
  }
}

PtoDirectReducePlan PlanPtoDirectReduce(const Call &call,
                                        arith::Analyzer *analyzer,
                                        const Map<Var, int> &buffer_versions) {
  PtoDirectReducePlan plan;
  plan.source = NormalizeToBufferRegion(call->args[0]);
  plan.destination = NormalizeToBufferRegion(call->args[1]);
  plan.operation = Downcast<StringImm>(call->args[2])->value;
  plan.clear = Downcast<IntImm>(call->args[4])->value != 0;
  const Buffer &src = plan.source->buffer;
  const Buffer &dst = plan.destination->buffer;
  plan.accumulator_dtype = dst->dtype;
  plan.source_prefix = PtoVersionPrefix(plan.source, analyzer, buffer_versions);
  plan.destination_prefix =
      PtoVersionPrefix(plan.destination, analyzer, buffer_versions);
  plan.logical_rank = static_cast<int>(src->shape.size()) - plan.source_prefix;
  plan.reduction_axis = Downcast<IntImm>(call->args[3])->value;
  plan.output_axis = plan.logical_rank == 1 ? 0 : 1 - plan.reduction_axis;
  plan.reduction_extent =
      Downcast<IntImm>(src->shape[plan.source_prefix + plan.reduction_axis])
          ->value;
  plan.output_extent =
      plan.logical_rank == 1
          ? 1
          : Downcast<IntImm>(src->shape[plan.source_prefix + plan.output_axis])
                ->value;
  bool keep_dims =
      static_cast<int>(dst->shape.size()) - plan.destination_prefix ==
      plan.logical_rank;
  plan.destination_output_axis =
      plan.destination_prefix +
      (keep_dims && plan.logical_rank == 2 ? plan.output_axis : 0);
  Array<PrimExpr> src_strides = PtoBufferStrides(src, analyzer);
  plan.reduce_across_lanes = analyzer->CanProveEqual(
      src_strides[plan.source_prefix + plan.reduction_axis], 1);
  return plan;
}

std::string AccessPatternToString(AccessPattern pattern) {
  switch (pattern) {
  case AccessPattern::kContinuous:
    return "continuous";
  case AccessPattern::kLaneUniform:
    return "lane-uniform";
  case AccessPattern::kIndexed:
    return "indexed";
  default:
    return "unknown";
  }
}

// SubstituteVar, ResolveIndexAliases and IsSupportedAddressExpr live in
// pto_index_analysis.cc; PTO lane-layout planning and the consumers here share
// that single implementation.

PtoInverseMapping InvertPtoLaneLayout(const For &loop) {
  auto anno = loop->annotations.Get(attr::kPtoParallelLoopLayout);
  if (!anno.has_value()) {
    throw PtoAnalysisError(
        "pto_parallel_loop_layout is missing on the outermost T.Parallel");
  }
  auto fragment_any = anno.value().try_cast<Fragment>();
  if (!fragment_any.has_value()) {
    throw PtoAnalysisError(
        "pto_parallel_loop_layout must be a tl.Fragment annotation");
  }
  Fragment fragment = fragment_any.value();

  // A 2D unit is the outermost For of a two-layer
  // adjacent Parallel nest. The annotation — and the fragment — lives on
  // the outer For and describes the full (i, j) input space; the input
  // coordinate order stays (i, j) whichever dimension is vectorized.
  const ForNode *inner_node = nullptr;
  if (const auto *body_for = loop->body.as<ForNode>()) {
    if (body_for->kind == ForKind::kParallel) {
      inner_node = body_for;
    }
  }
  const bool is_2d = inner_node != nullptr;
  const int expected_dim = is_2d ? 2 : 1;
  if (fragment->InputDim() != expected_dim) {
    std::ostringstream oss;
    oss << "pto_parallel_loop_layout input_size must be " << expected_dim
        << "D for a "
        << (is_2d ? "two-layer nested T.Parallel" : "1D T.Parallel")
        << ", got input dim " << fragment->InputDim();
    throw PtoAnalysisError(oss.str());
  }
  if (fragment->GetForwardIndex().size() != 1) {
    // Extra per-lane position dimensions are a legal-but-unsupported
    // layout shape; reject with a layout diagnostic instead of an internal
    // assertion inside Forward.
    std::ostringstream oss;
    oss << "pto_parallel_loop_layout must have exactly one internal "
           "position dimension, got "
        << fragment->GetForwardIndex().size();
    throw PtoAnalysisError(oss.str());
  }
  if (!is_one(fragment->ReplicateExtent())) {
    throw PtoAnalysisError(
        "pto_parallel_loop_layout requires replicate_size == 1");
  }

  PrimExpr thread_extent = fragment->ThreadExtent();
  const int64_t *lanes_ptr = as_const_int(thread_extent);
  if (lanes_ptr == nullptr || *lanes_ptr <= 0) {
    throw PtoAnalysisError(
        "pto_parallel_loop_layout thread extent must be a positive constant");
  }
  int64_t lanes = *lanes_ptr;
  if (lanes != 64 && lanes != 128 && lanes != 256) {
    std::ostringstream oss;
    oss << "T.SimdVF lanes must be one of {64, 128, 256}, got " << lanes;
    throw PtoAnalysisError(oss.str());
  }

  // Shape contract: the fragment input shape must match the loop extents
  // (1D: the single extent; 2D: (outer, inner)).
  const int64_t *extent_ptr = as_const_int(loop->extent);
  if (extent_ptr == nullptr) {
    // The helper must not assert on inputs the callers did not
    // pre-validate; a symbolic extent is a layout diagnostic (same rule
    // as the inner extent below).
    std::ostringstream oss;
    oss << "pto_parallel_loop_layout loop extent must be a compile-time "
           "constant, got "
        << loop->extent;
    throw PtoAnalysisError(oss.str());
  }
  int64_t outer_extent = *extent_ptr;
  int64_t inner_extent = 1;
  if (is_2d) {
    const int64_t *inner_extent_ptr = as_const_int(inner_node->extent);
    if (inner_extent_ptr == nullptr) {
      // The helper must not assert on inputs the callers did not
      // pre-validate; a symbolic inner extent is a layout diagnostic.
      std::ostringstream oss;
      oss << "pto_parallel_loop_layout inner loop extent must be a "
             "compile-time constant, got "
          << inner_node->extent;
      throw PtoAnalysisError(oss.str());
    }
    inner_extent = *inner_extent_ptr;
  }
  // Shape contract: the fragment input space is padded on the
  // vectorized dimension. 1D: the single input extent must be the padded
  // extent P = ceil(E / L) * L. 2D: the kept dimension keeps its original
  // loop extent and the selected dimension is padded to P — but which
  // dimension is selected is only known jointly with the canonical mapping
  // after the inversion, so the 2D shape check runs post-inversion. Here
  // the 1D case is checked directly; an unpadded [E] (E % L != 0), an
  // over-padded extent, and (for divisible E) any extent != E are layout
  // diagnostics.
  Array<PrimExpr> input_shape = fragment->InputShape();
  if (!is_2d) {
    if (input_shape.size() != 1u) {
      std::ostringstream oss;
      oss << "pto_parallel_loop_layout input shape " << input_shape
          << " must be 1D for a 1D T.Parallel";
      throw PtoAnalysisError(oss.str());
    }
    const auto [pad_1d, padded_1d] = CheckedPtoPadding(outer_extent, lanes);
    (void)pad_1d;
    const int64_t *frag_extent = as_const_int(input_shape[0]);
    if (frag_extent == nullptr || *frag_extent != padded_1d) {
      std::ostringstream oss;
      oss << "pto_parallel_loop_layout input shape " << input_shape
          << " does not match the padded extent " << padded_1d
          << " (loop extent " << outer_extent << ", lanes " << lanes << "); "
          << "the vectorized dimension must be padded to ceil(E/L)*L";
      throw PtoAnalysisError(oss.str());
    }
  } else {
    if (input_shape.size() != 2u) {
      std::ostringstream oss;
      oss << "pto_parallel_loop_layout input shape " << input_shape
          << " must be 2D for a two-layer nested T.Parallel";
      throw PtoAnalysisError(oss.str());
    }
    // Structural presence check only; the per-dimension contract (kept =
    // original, selected = padded) is a joint check with the canonical
    // mapping below.
    for (const auto &dim : input_shape) {
      if (as_const_int(dim) == nullptr) {
        std::ostringstream oss;
        oss << "pto_parallel_loop_layout input shape " << input_shape
            << " must be compile-time constants, got " << input_shape;
        throw PtoAnalysisError(oss.str());
      }
    }
  }

  // Thread-range contract: [0, L).
  Range thread_range = fragment->ThreadRange();
  if (thread_range.defined()) {
    const int64_t *tr_min = as_const_int(thread_range->min);
    const int64_t *tr_ext = as_const_int(thread_range->extent);
    if (tr_min == nullptr || *tr_min != 0 || tr_ext == nullptr ||
        *tr_ext != lanes) {
      std::ostringstream oss;
      oss << "pto_parallel_loop_layout thread_range must be [0, " << lanes
          << "), got [" << thread_range->min << ", " << thread_range->extent
          << ")";
      throw PtoAnalysisError(oss.str());
    }
  }

  // The fragment is built over int32 placeholder vars (the only
  // dtype LayoutInference produces); a loop var of any other dtype (e.g.
  // int64) crashes the tvm substitution machinery with a raw assertion
  // during inversion. Reject it here with a layout diagnostic instead.
  if (loop->loop_var.dtype() != DataType::Int(32)) {
    std::ostringstream oss;
    oss << "pto_parallel_loop_layout requires an int32 loop variable (the "
           "dtype LayoutInference produces), got "
        << loop->loop_var.dtype();
    throw PtoAnalysisError(oss.str());
  }
  if (is_2d && inner_node->loop_var.dtype() != DataType::Int(32)) {
    std::ostringstream oss;
    oss << "pto_parallel_loop_layout requires an int32 inner loop variable "
           "(the dtype LayoutInference produces), got "
        << inner_node->loop_var.dtype();
    throw PtoAnalysisError(oss.str());
  }

  // Invert through the real inverse layout (single source of truth for the
  // mapping). InverseWithLevel appends the lane (thread) input after the
  // per-lane position input. It raises when the fragment is not a valid
  // bijection — e.g. a hand-built layout whose per-thread position extent
  // does not cover the input space: Layout::InverseWithLevel throws
  // NormalizeIterException when the iteration map has errors, and the
  // iteration-map machinery itself reports tvm errors as ffi::Error. Both are
  // known input failures here, so they become layout diagnostics. Only this
  // call is wrapped, so a genuine implementation error anywhere else still
  // surfaces as one instead of being reported as bad input.
  auto invert = [&]() {
    try {
      return fragment->InverseWithLevel(/*require_padding_guard=*/false);
    } catch (const NormalizeIterException &err) {
      throw PtoAnalysisError(std::string("pto_parallel_loop_layout cannot be "
                                         "inverted: ") +
                             err.what());
    } catch (const ffi::Error &err) {
      throw PtoAnalysisError(std::string("pto_parallel_loop_layout cannot be "
                                         "inverted: ") +
                             err.what());
    }
  };
  auto inverse_info = invert();
  if (inverse_info.second != arith::IterMapLevel::Bijective) {
    std::ostringstream oss;
    oss << "pto_parallel_loop_layout is not bijectively invertible (level "
        << static_cast<int>(inverse_info.second)
        << "): " << fragment->DebugOutput();
    throw PtoAnalysisError(oss.str());
  }
  Layout inverse = inverse_info.first;

  // Forward (chunk, lane) through the inverse layout to obtain the logical
  // iteration coordinates followed by the replicate coordinate. With
  // replicate_size == 1, the trailing replicate coordinate is zero.
  // The branches below use the logical coordinates for the loop dimensionality.
  Var chunk_var("pto_chunk", loop->loop_var.dtype());
  Var lane_var("pto_lane", loop->loop_var.dtype());
  Array<PrimExpr> logical = inverse->Forward({chunk_var, lane_var});
  // The output count follows the input dimensionality (1D: (i, replicate);
  // 2D: (i, j, replicate)); each branch below checks its own arity.

  // Only the canonical mappings are accepted. Any other
  // invertible form (e.g. a reversed lane assignment) is rejected instead
  // of being silently replaced with our own formula.
  //   1D:      i = chunk * L + lane
  //   2D (j):  j = (chunk * L + lane) % N, i = (chunk * L + lane) / N
  //   2D (i):  i = (chunk * L + lane) % M, j = (chunk * L + lane) / M
  PtoInverseMapping mapping;
  mapping.lane_var = lane_var;
  mapping.chunk_var = chunk_var;
  mapping.lanes = lanes;

  if (!is_2d) {
    ICHECK_EQ(logical.size(), 2u);
    // The fragment input space is padded to P, so the chunk
    // position range is [0, P/L) — the padded chunk count Q.
    const int64_t padded = CheckedPtoPadding(outer_extent, lanes).second;
    arith::Analyzer probe;
    probe.Bind(chunk_var,
               Range::FromMinExtent(make_zero(chunk_var->dtype),
                                    IntImm(chunk_var->dtype, padded / lanes)));
    probe.Bind(lane_var, Range::FromMinExtent(make_zero(lane_var->dtype),
                                              IntImm(lane_var->dtype, lanes)));
    PrimExpr canonical = chunk_var * IntImm(chunk_var->dtype, lanes) + lane_var;
    if (!probe.CanProveEqual(logical[0], canonical)) {
      std::ostringstream oss;
      oss << "pto_parallel_loop_layout does not use the canonical 1D "
             "mapping i = chunk * L + lane (got "
          << logical[0] << " from " << fragment->DebugOutput() << ")";
      throw PtoAnalysisError(oss.str());
    }
    mapping.logical_extent = outer_extent;
    mapping.padded_extent = padded;
    mapping.chunk_count = padded / lanes;
    mapping.tail_lanes = outer_extent % lanes;
    mapping.index_expr = logical[0];
    return mapping;
  }

  ICHECK_EQ(logical.size(), 3u);
  const int64_t M = outer_extent;
  const int64_t N = inner_extent;
  // The fragment input space is padded on the vectorized dimension, so
  // the flattened position t = kept * P_sel + sel lives in [0, K * P_sel)
  // and the global chunk count is K * ceil(E / L) (each row padded
  // independently — not ceil(K * E / L), which is not equivalent). This
  // probe domain is for the canonical mapping check only; the
  // address-proof domain that AnalyzeBufferAccess uses binds the per-row
  // q range instead.
  // Each interpretation gets its own probe with the exact chunk domain
  // [0, kept * P_sel / L): a domain bound larger than the fragment's real
  // chunk count could make an identity that holds on the real domain
  // unprovable, and a smaller one could admit a mapping that only looks
  // canonical on part of the space. The lane range [0, L) is shared.
  const auto bind_probe = [&](arith::Analyzer *probe, int64_t chunk_bound) {
    probe->Bind(chunk_var,
                Range::FromMinExtent(make_zero(chunk_var->dtype),
                                     IntImm(chunk_var->dtype, chunk_bound)));
    probe->Bind(lane_var, Range::FromMinExtent(make_zero(lane_var->dtype),
                                               IntImm(lane_var->dtype, lanes)));
  };
  PrimExpr t = chunk_var * IntImm(chunk_var->dtype, lanes) + lane_var;
  // The vectorized coordinate is t % P_sel, the kept one t / P_sel — both
  // the modulus and the divisor use the *padded* selected extent. The
  // interpretation is accepted only when it also agrees with the fragment
  // input shape: the kept dimension keeps its original loop extent and
  // the selected dimension is padded up to P_sel (unpadded / over-padded
  // / wrongly-modified kept dims each fail here). A degenerate dimension
  // (extent 1) makes both canonical forms provable; the shape agreement
  // breaks the tie exactly when only one dim was padded, and the inner
  // preference remains for fully divisible shapes.
  const auto matches = [&](bool select_inner) -> bool {
    const int64_t sel_extent = select_inner ? N : M;
    const auto [pad, padded] = CheckedPtoPadding(sel_extent, lanes);
    const int64_t kept_extent = select_inner ? M : N;
    const int64_t *frag_kept = as_const_int(input_shape[select_inner ? 0 : 1]);
    const int64_t *frag_sel = as_const_int(input_shape[select_inner ? 1 : 0]);
    if (frag_kept == nullptr || frag_sel == nullptr ||
        *frag_kept != kept_extent || *frag_sel != padded) {
      return false;
    }
    (void)pad;
    arith::Analyzer probe;
    bind_probe(&probe, kept_extent * padded / lanes);
    const PrimExpr &sel = select_inner ? logical[1] : logical[0];
    const PrimExpr &kept = select_inner ? logical[0] : logical[1];
    PrimExpr sel_imm = IntImm(t.dtype(), padded);
    return probe.CanProveEqual(sel, FloorMod(t, sel_imm)) &&
           probe.CanProveEqual(kept, FloorDiv(t, sel_imm));
  };
  const bool inner_ok = matches(/*select_inner=*/true);
  const bool outer_ok = matches(/*select_inner=*/false);
  if (!inner_ok && !outer_ok) {
    std::ostringstream oss;
    oss << "pto_parallel_loop_layout does not use a canonical 2D mapping "
           "(j = t % P_N, i = t / P_N  or  i = t % P_M, j = t / P_M with "
           "t = chunk * L + lane, P_* = padded selected extent; the kept "
           "dim keeps its loop extent in the fragment shape); got (i, j) = ("
        << logical[0] << ", " << logical[1] << "), shape " << input_shape
        << ", loop extents (" << M << ", " << N << ") from "
        << fragment->DebugOutput();
    throw PtoAnalysisError(oss.str());
  }
  // Both dims matching is degenerate (e.g. M == N == 1); prefer the inner
  // dimension, consistent with the layout-selection rule.
  const bool select_inner = inner_ok;
  const int64_t sel_extent = select_inner ? N : M;
  const auto [pad_sel, P_sel] = CheckedPtoPadding(sel_extent, lanes);
  (void)pad_sel;
  mapping.is_2d = true;
  mapping.outer_var = loop->loop_var;
  mapping.inner_var = inner_node->loop_var;
  mapping.outer_extent = M;
  mapping.inner_extent = N;
  mapping.select_inner = select_inner;
  mapping.logical_extent = sel_extent;
  mapping.padded_extent = P_sel;
  mapping.chunk_count = P_sel / lanes;
  mapping.tail_lanes = sel_extent % lanes;
  mapping.index_expr = t;
  return mapping;
}

MemoryAccess AnalyzeBufferAccess(const Buffer &buffer,
                                 const Array<PrimExpr> &indices, bool is_write,
                                 const Var &loop_var,
                                 const PtoInverseMapping &mapping,
                                 arith::Analyzer *analyzer,
                                 const Map<Var, PrimExpr> &bind_env) {
  MemoryAccess access;
  access.buffer = buffer;
  access.elem_dtype = buffer->dtype;
  access.is_write = is_write;

  Optional<Var> kept_var;
  int64_t kept_extent = 0;
  if (mapping.is_2d && mapping.serial_var() != nullptr) {
    kept_var = GetRef<Var>(mapping.serial_var());
    kept_extent =
        mapping.select_inner ? mapping.outer_extent : mapping.inner_extent;
  }
  PtoAddressAnalysis analyzed = AnalyzePtoAddress(
      buffer, indices, loop_var, mapping.index_expr, mapping.lane_var,
      mapping.chunk_var, mapping.lanes, mapping.chunk_count, kept_var,
      kept_extent, bind_env, analyzer);
  access.original_offset = analyzed.original_offset;
  access.scalar_base_offset = analyzed.scalar_base_offset;
  access.relative_extent = analyzed.relative_extent;
  access.source_indices = analyzed.source_indices;
  access.indices_after_inverse = analyzed.indices_after_inverse;
  access.start_indices = analyzed.start_indices;
  switch (analyzed.pattern) {
  case PtoAddressPattern::kContinuous:
    access.pattern = AccessPattern::kContinuous;
    break;
  case PtoAddressPattern::kLaneUniform:
    access.pattern = AccessPattern::kLaneUniform;
    break;
  case PtoAddressPattern::kIndexed:
    access.pattern = AccessPattern::kIndexed;
    break;
  }
  return access;
}

PrimExpr PreparePtoIndexedOffset(const MemoryAccess &access,
                                 const PrimExpr &relative) {
  PrimExpr native = relative;
  while (const auto *conversion = native.as<CastNode>()) {
    const DataType from = conversion->value.dtype();
    const DataType to = conversion->dtype;
    if (!(from.is_int() || from.is_uint()) || !(to.is_int() || to.is_uint()) ||
        from.bits() >= to.bits() || !conversion->annotations.empty()) {
      break;
    }
    native = conversion->value;
  }
  PrimExpr result = relative;
  if (native.dtype().bits() == 16) {
    for (const PrimExpr &index : access.source_indices) {
      if (ExprDeepEqual()(native, index)) {
        result = native;
        break;
      }
    }
  }
  if (result.dtype().bits() > access.elem_dtype.bits()) {
    std::ostringstream oss;
    oss << "indexed " << result.dtype().bits()
        << "-bit offsets require implicit narrowing to "
        << access.elem_dtype.bits()
        << " bits; use a native 16-bit index or an explicit source Cast only "
           "after ensuring the complete element offset is representable";
    throw PtoAnalysisError(oss.str());
  }
  return result;
}

namespace {

// Only coordinates and their integer arithmetic can be generated at a narrower
// width without converting an already loaded vector. Check every intermediate
// on active iterations. Tail offsets are sanitized before indexed memory
// operations, so integer wrap in an inactive lane cannot affect an access.
bool CanGenerateUint16Offsets(const PrimExpr &offset, const Var &loop_var,
                              const PtoInverseMapping &mapping,
                              arith::Analyzer *analyzer) {
  std::unique_ptr<arith::Analyzer> proof = analyzer->Clone();
  proof->Bind(loop_var, Range::FromMinExtent(
                            make_zero(loop_var.dtype()),
                            IntImm(loop_var.dtype(), mapping.logical_extent)));
  Optional<Var> kept;
  if (mapping.is_2d) {
    kept = mapping.select_inner ? mapping.outer_var : mapping.inner_var;
    int64_t extent =
        mapping.select_inner ? mapping.outer_extent : mapping.inner_extent;
    proof->Bind(kept.value(),
                Range::FromMinExtent(make_zero(kept.value().dtype()),
                                     IntImm(kept.value().dtype(), extent)));
  }
  std::function<bool(const PrimExpr &)> check = [&](const PrimExpr &expr) {
    if (expr.dtype() != DataType::Int(32) &&
        expr.dtype() != DataType::UInt(32)) {
      return false;
    }
    if (const auto *add = expr.as<AddNode>()) {
      if (!check(add->a) || !check(add->b)) {
        return false;
      }
    } else if (const auto *sub = expr.as<SubNode>()) {
      if (!check(sub->a) || !check(sub->b)) {
        return false;
      }
    } else if (const auto *mul = expr.as<MulNode>()) {
      if (!check(mul->a) || !check(mul->b)) {
        return false;
      }
    } else if (!expr.as<IntImmNode>() && !expr.as<VarNode>()) {
      return false;
    }
    arith::ConstIntBound bound = proof->const_int_bound(expr);
    return bound->min_value >= 0 && bound->max_value <= 65535;
  };
  return check(offset);
}

} // namespace

PtoIndexedAccessPlan PlanPtoIndexedAccess(const MemoryAccess &access,
                                          const IndexedMemoryOwner &owner,
                                          const Var &loop_var,
                                          const PtoInverseMapping &mapping,
                                          arith::Analyzer *analyzer) {
  PtoIndexedAccessPlan plan{make_zero(DataType::Int(32)),
                            access.original_offset, false};
  // A large allocation can still be addressed directly when every active
  // coordinate offset fits ui16. Its unused capacity need not fit the window.
  if (access.elem_dtype.bits() == 16 &&
      CanGenerateUint16Offsets(plan.offsets, loop_var, mapping, analyzer)) {
    plan.uint16_offsets = true;
    return plan;
  }
  if (access.elem_dtype.bits() == 16 &&
      !analyzer->CanProve(owner.extent <= 65536)) {
    plan.rebased = true;
    plan.base = access.scalar_base_offset;
    if (!analyzer->CanProve(access.relative_extent > 0) ||
        !analyzer->CanProve(access.relative_extent <= 65536)) {
      throw PtoAnalysisError("B16 indexed access requires a ui16 relative "
                             "window of at most 65536 elements");
    }
    // Keep emitted expressions independent of the caller's simplification
    // facts, so Verify and Vectorize prepare the same relative offset.
    arith::Analyzer expression_analyzer;
    plan.offsets = expression_analyzer.Simplify(access.original_offset -
                                                access.scalar_base_offset);
  }
  if (access.elem_dtype.bits() == 16 && plan.offsets.dtype().bits() == 32 &&
      CanGenerateUint16Offsets(plan.offsets, loop_var, mapping, analyzer)) {
    plan.uint16_offsets = true;
  } else {
    plan.offsets = PreparePtoIndexedOffset(access, plan.offsets);
  }
  return plan;
}

void CheckPtoIndexedAccessPlan(const PtoIndexedAccessPlan &plan,
                               const IndexedMemoryOwner &owner,
                               const Var &loop_var,
                               const PtoInverseMapping &mapping,
                               arith::Analyzer *analyzer) {
  if (!plan.rebased) {
    return;
  }
  std::unique_ptr<arith::Analyzer> proof = analyzer->Clone();
  proof->Bind(mapping.chunk_var,
              Range::FromMinExtent(
                  make_zero(mapping.chunk_var.dtype()),
                  IntImm(mapping.chunk_var.dtype(), mapping.chunk_count)));
  if (mapping.is_2d) {
    const Var kept =
        mapping.select_inner ? mapping.outer_var : mapping.inner_var;
    const int64_t extent =
        mapping.select_inner ? mapping.outer_extent : mapping.inner_extent;
    proof->Bind(kept, Range::FromMinExtent(make_zero(kept.dtype()),
                                           IntImm(kept.dtype(), extent)));
  }
  PrimExpr base = SubstituteVar(plan.base, loop_var.get(), mapping.index_expr);
  base = proof->Simplify(SubstituteVar(base, mapping.lane_var.get(),
                                       make_zero(mapping.lane_var.dtype())));
  if (!proof->CanProve(base >= 0) || !proof->CanProve(base < owner.extent)) {
    std::ostringstream oss;
    oss << "B16 indexed access requires the scalar instruction base D to be "
           "provably inside the owner (0 <= D < "
        << owner.extent << "), but the rebased access produces base " << base;
    throw PtoAnalysisError(oss.str());
  }
}

bool IsSupportedElementDType(DataType dtype) {
  // No packed sub-byte element types (FP4 etc.), no
  // buffer-element dtypes that already carry vector lanes, and no FP64
  // (the current PTO codegen has no FP64 scalar-type mapping).
  if (dtype.bits() < 8 || dtype.lanes() != 1) {
    // FP4 (e2m1) and every other sub-byte element type land here: an FP4
    // element is 4 bits wide whether packed or not, so a separate FP4 check
    // would be unreachable.
    return false;
  }
  // The PTO codegen only maps 8/16/32-bit scalars; unusual widths (int24) and
  // FP64 (no scalar-type mapping) are rejected for buffers and arithmetic
  // alike by the width rule below, not just inside Cast planning.
  if (!dtype.is_bool() && dtype.bits() != 8 && dtype.bits() != 16 &&
      dtype.bits() != 32) {
    return false;
  }
  return dtype.is_float() || dtype.is_bfloat16() || dtype.is_int() ||
         dtype.is_uint() || dtype.is_bool() || IsSupportedFloat8(dtype);
}

bool IsSupportedFloat8(DataType dtype) {
  // Only these two take part in the FP8 conversion matrix. float8_e4m3 is a
  // distinct TIR dtype from float8_e4m3fn and is not admitted by name
  // similarity; E8M0/HiF8/FNUZ are outside the supported conversion range,
  // while FP4 is a packed sub-byte type with no unified element addressing.
  return dtype.is_float8_e4m3fn() || dtype.is_float8_e5m2();
}

namespace {

// Whitelist check for *value* expressions. A
// BufferLoad's indices belong to address analysis, not value lowering
// the version-dim FloorMod/FloorDiv inside a load index must not
// be rejected here. The value whitelist does not descend into indices.
class WhitelistVisitor : public ExprVisitor {
public:
  std::optional<std::string> unsupported;

  void VisitExpr_(const BufferLoadNode *) final {}
  void VisitExpr_(const VarNode *) final {}
  void VisitExpr_(const IntImmNode *) final {}
  void VisitExpr_(const FloatImmNode *) final {}
  void VisitExpr_(const StringImmNode *) final {
    // String constants are metadata markers only. The base
    // ExprVisitor declares StringImm and accepts it as a leaf, so the
    // VisitExprDefault_ fallback never fired and `Evaluate(StringImm)`
    // slipped through Verify into Vectorize's internal error. Reject it
    // explicitly here, with the same wording as the lane-use analysis.
    if (!unsupported.has_value()) {
      unsupported = "string constants are metadata-only; they cannot "
                    "appear in computation values";
    }
  }
  void VisitExpr_(const AddNode *op) final { ExprVisitor::VisitExpr_(op); }
  void VisitExpr_(const SubNode *op) final { ExprVisitor::VisitExpr_(op); }
  void VisitExpr_(const MulNode *op) final { ExprVisitor::VisitExpr_(op); }
  void VisitExpr_(const DivNode *op) final { ExprVisitor::VisitExpr_(op); }
  void VisitExpr_(const CastNode *op) final {
    // Only the converted value is a computation value. The base visitor also
    // walks PrimExpr entries of the cast annotations, which would put a
    // StringImm rounding hint in front of the value whitelist and reject an
    // otherwise legal cast ("string constants are metadata-only"). Attribute
    // legality is decided by PlanValueCast, from the raw annotations.
    VisitExpr(op->value);
  }
  // The base ExprVisitor overrides VisitExpr_ for every standard node and
  // silently descends, so non-whitelist nodes never reach
  // VisitExprDefault_. Reject them explicitly.
  void MarkUnsupported(const char *kind) {
    if (!unsupported.has_value()) {
      unsupported = std::string("unsupported expression node `") + kind +
                    "` (only buffer loads, constants, "
                    "scalars, loop indices, Add, Sub, Mul, Div, "
                    "fabs/exp/log/sqrt and Cast)";
    }
  }
  void VisitExpr_(const ModNode *) final { MarkUnsupported("Mod"); }
  void VisitExpr_(const FloorDivNode *) final { MarkUnsupported("FloorDiv"); }
  void VisitExpr_(const FloorModNode *) final { MarkUnsupported("FloorMod"); }
  void VisitExpr_(const MinNode *) final { MarkUnsupported("Min"); }
  void VisitExpr_(const MaxNode *) final { MarkUnsupported("Max"); }
  void VisitExpr_(const EQNode *) final { MarkUnsupported("EQ"); }
  void VisitExpr_(const NENode *) final { MarkUnsupported("NE"); }
  void VisitExpr_(const LTNode *) final { MarkUnsupported("LT"); }
  void VisitExpr_(const LENode *) final { MarkUnsupported("LE"); }
  void VisitExpr_(const GTNode *) final { MarkUnsupported("GT"); }
  void VisitExpr_(const GENode *) final { MarkUnsupported("GE"); }
  void VisitExpr_(const AndNode *) final { MarkUnsupported("And"); }
  void VisitExpr_(const OrNode *) final { MarkUnsupported("Or"); }
  void VisitExpr_(const NotNode *) final { MarkUnsupported("Not"); }
  void VisitExpr_(const SelectNode *) final {
    if (!unsupported.has_value()) {
      unsupported =
          "Select is not vectorizable yet (integer T.abs lowers to Select; "
          "it arrives with the PR-2 Select/cmp support)";
    }
  }
  void VisitExpr_(const RampNode *) final { MarkUnsupported("Ramp"); }
  void VisitExpr_(const BroadcastNode *) final { MarkUnsupported("Broadcast"); }
  void VisitExpr_(const ShuffleNode *) final { MarkUnsupported("Shuffle"); }
  void VisitExpr_(const ReduceNode *) final { MarkUnsupported("Reduce"); }
  void VisitExpr_(const CallNode *op) final {
    if (auto call_op = op->op.as<Op>()) {
      const std::string &name = call_op.value()->name;
      // Hardware-backed unary math. fabs/exp/log/sqrt map to the VMI
      // vabs/vexp/vln/vsqrt instructions; the dtype matrix mirrors the
      // ISA documentation (see IsSupportedVectorUnaryOp).
      if (name == "tirx.fabs" || name == "tirx.exp" || name == "tirx.log" ||
          name == "tirx.sqrt") {
        if (op->args.size() != 1U) {
          if (!unsupported.has_value()) {
            unsupported = name + " expects exactly one operand";
          }
          return;
        }
        const DataType arg = op->args[0].dtype();
        if (!IsSupportedVectorUnaryOp(name, arg)) {
          if (!unsupported.has_value()) {
            std::ostringstream oss;
            oss << arg;
            unsupported = name + " vectorization does not support " +
                          oss.str() +
                          " operands (per the VMI operation matrix)";
          }
          return;
        }
        ExprVisitor::VisitExpr_(op);
        return;
      }
    }
    {
      std::string name = "call";
      if (const auto *op_node = op->op.as<OpNode>()) {
        name = op_node->name;
      }
      if (!unsupported.has_value()) {
        unsupported = "unsupported Call `" + name +
                      "` (only tirx.fabs/exp/log/sqrt are vectorizable)";
      }
    }
  }

  void VisitExprDefault_(const ffi::Object *op) final {
    std::string key = op->GetTypeKey();
    MarkUnsupported(key.c_str());
  }
};

} // namespace

// ---------------------------------------------------------------------------
// Shared SIMD_VF region scan: one implementation for both passes.
// ---------------------------------------------------------------------------
bool UsesExplicitSIMD(const Stmt &stmt) {
  bool found = false;
  PostOrderVisit(stmt, [&](const ObjectRef &node) {
    if (const auto *call = node.as<CallNode>()) {
      if (const auto *op = call->op.as<OpNode>()) {
        found |= std::string(op->name).rfind("tl.simd.", 0) == 0;
      }
    }
  });
  return found;
}

namespace {

void CollectAllParallelDeep(const Stmt &stmt,
                            std::vector<const ForNode *> &out) {
  if (const auto *for_node = stmt.as<ForNode>()) {
    if (for_node->kind == ForKind::kParallel) {
      if (UsesExplicitSIMD(for_node->body)) {
        // The explicit nest itself needs no conversion, but an automatic
        // Parallel below it must still be reported as unsupported.
        CollectAllParallelDeep(for_node->body, out);
        return;
      }
      out.push_back(for_node);
    }
    CollectAllParallelDeep(for_node->body, out);
    return;
  }
  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    for (const auto &s : seq->seq) {
      CollectAllParallelDeep(s, out);
    }
    return;
  }
  if (const auto *attr = stmt.as<AttrStmtNode>()) {
    CollectAllParallelDeep(attr->body, out);
    return;
  }
  if (const auto *block = stmt.as<SBlockNode>()) {
    CollectAllParallelDeep(block->body, out);
    return;
  }
  if (const auto *realize = stmt.as<SBlockRealizeNode>()) {
    CollectAllParallelDeep(realize->block->body, out);
    return;
  }
  if (const auto *ite = stmt.as<IfThenElseNode>()) {
    CollectAllParallelDeep(ite->then_case, out);
    if (ite->else_case.defined()) {
      CollectAllParallelDeep(ite->else_case.value(), out);
    }
    return;
  }
  if (const auto *while_node = stmt.as<WhileNode>()) {
    CollectAllParallelDeep(while_node->body, out);
    return;
  }
}

void DiscoverUnitsShared(const Stmt &stmt, PtoRegionScan *scan) {
  if (const auto *for_node = stmt.as<ForNode>()) {
    if (for_node->kind == ForKind::kParallel) {
      if (UsesExplicitSIMD(for_node->body)) {
        std::vector<const ForNode *> under;
        CollectAllParallelDeep(for_node->body, under);
        if (!under.empty()) {
          scan->has_unsupported_wrapper = true;
          scan->issue = "an automatic T.Parallel nested inside an explicit "
                        "SIMD T.Parallel is not yet supported";
        }
        return;
      }
      // A two-layer adjacent Parallel nest is one
      // 2D unit; the layout annotation lives on the outermost For. A
      // third Parallel layer (or a Parallel deeper inside) stays
      // unsupported.
      const auto *adjacent_inner = for_node->body.as<ForNode>();
      const bool inner_is_parallel = adjacent_inner != nullptr &&
                                     adjacent_inner->kind == ForKind::kParallel;
      std::vector<const ForNode *> deeper;
      CollectAllParallelDeep(
          inner_is_parallel ? adjacent_inner->body : for_node->body, deeper);
      if (!deeper.empty()) {
        scan->has_unsupported_wrapper = true;
        // Two distinct illegal shapes: a genuine third parallel layer
        // inside an adjacent two-layer nest, or an outer Parallel whose
        // body mixes other statements with an inner Parallel (the nest is
        // not adjacent and cannot form a pure 2D unit).
        scan->issue =
            inner_is_parallel
                ? "three or more nested parallel layers are not yet "
                  "supported"
                : "a T.Parallel whose body mixes other statements with an "
                  "inner T.Parallel (a non-adjacent nest) is not yet "
                  "supported";
      }
      scan->units.push_back(GetRef<For>(for_node));
      return;
    }
    if (for_node->kind == ForKind::kVectorized ||
        for_node->kind == ForKind::kThreadBinding) {
      // Only sequential wrappers may surround a unit: a re-vectorizing loop
      // contradicts the vectorization this conversion already performs, and
      // a thread binding contradicts the no-thread-binding contract of the
      // converting region. A region without T.Parallel under them stays a
      // legitimate hand-written VMI region and is skipped entirely.
      std::vector<const ForNode *> under;
      CollectAllParallelDeep(stmt, under);
      if (!under.empty()) {
        scan->has_unsupported_wrapper = true;
        if (for_node->kind == ForKind::kThreadBinding) {
          std::ostringstream oss;
          oss << "a thread-bound loop (ForKind::kThreadBinding";
          if (const auto *iv = for_node->thread_binding.as<IterVarNode>()) {
            oss << ", thread_binding `" << iv->var->name_hint << "`";
          }
          oss << ") around T.Parallel is not supported: "
                 "thread bindings are not allowed inside a converting "
                 "SIMD_VF block";
          scan->issue = oss.str();
        } else {
          scan->issue =
              "a vectorized loop (ForKind::kVectorized) around T.Parallel is "
              "not yet supported";
        }
      }
      return;
    }
    // A serial loop (with or without an unroll hint) is a sequential
    // wrapper: its iterations run in order and each iteration keeps the
    // unit's semantics, so it stays accepted.
    DiscoverUnitsShared(for_node->body, scan);
    return;
  }
  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    for (const auto &s : seq->seq) {
      DiscoverUnitsShared(s, scan);
    }
    return;
  }
  if (const auto *attr = stmt.as<AttrStmtNode>()) {
    DiscoverUnitsShared(attr->body, scan);
    return;
  }
  if (const auto *block = stmt.as<SBlockNode>()) {
    DiscoverUnitsShared(block->body, scan);
    return;
  }
  if (const auto *realize = stmt.as<SBlockRealizeNode>()) {
    DiscoverUnitsShared(realize->block->body, scan);
    return;
  }
  if (stmt.as<IfThenElseNode>() != nullptr || stmt.as<WhileNode>() != nullptr) {
    // Control-flow wrappers: a Parallel under them is rejected; without
    // one the region may still be a legitimate hand-written VMI region.
    std::vector<const ForNode *> under;
    CollectAllParallelDeep(stmt, under);
    if (!under.empty()) {
      scan->has_unsupported_wrapper = true;
      scan->issue = stmt.as<IfThenElseNode>() != nullptr
                        ? "T.Parallel under a conditional branch is not yet "
                          "supported"
                        : "T.Parallel under a while loop is not yet "
                          "supported";
    }
    return;
  }
}

} // namespace

PtoRegionScan ScanPtoRegion(const Stmt &region) {
  PtoRegionScan scan;
  DiscoverUnitsShared(region, &scan);
  class TileOpScanner : public StmtExprVisitor {
  public:
    explicit TileOpScanner(PtoRegionScan *scan) : scan_(scan) {}
    void VisitStmt_(const AttrStmtNode *op) final {
      if (auto node = op->node.try_cast<PrimExpr>()) {
        VisitExpr(node.value());
      }
      StmtExprVisitor::VisitStmt_(op);
    }
    void VisitStmt_(const BufferStoreNode *op) final {
      if (op->predicate.defined()) {
        VisitExpr(op->predicate.value());
      }
      StmtExprVisitor::VisitStmt_(op);
    }
    void VisitExpr_(const BufferLoadNode *op) final {
      if (op->predicate.defined()) {
        VisitExpr(op->predicate.value());
      }
      StmtExprVisitor::VisitExpr_(op);
    }
    void VisitStmt_(const SBlockNode *op) final {
      StmtExprVisitor::VisitStmt_(op);
      // The scan starts at the enclosing SIMD_VF's body. Another VF on
      // this lexical path would be checked with the outer lanes and then
      // lowered with its own lanes, so it cannot be a transparent wrapper.
      if (op->name_hint == "SIMD_VF") {
        scan_->has_unsupported_wrapper = true;
        scan_->issue = "nested SIMD_VF blocks are not supported; use "
                       "sequential T.SimdVF regions";
      }
    }
    void VisitStmt_(const ForNode *op) final {
      scan_->has_any_parallel |=
          op->kind == ForKind::kParallel && !UsesExplicitSIMD(op->body);
      int saved = forbidden_depth_;
      if (op->kind != ForKind::kSerial && op->kind != ForKind::kUnrolled) {
        ++forbidden_depth_;
      }
      StmtExprVisitor::VisitStmt_(op);
      forbidden_depth_ = saved;
    }
    void VisitStmt_(const IfThenElseNode *op) final {
      ++forbidden_depth_;
      StmtExprVisitor::VisitStmt_(op);
      --forbidden_depth_;
    }
    void VisitStmt_(const WhileNode *op) final {
      ++forbidden_depth_;
      StmtExprVisitor::VisitStmt_(op);
      --forbidden_depth_;
    }
    void VisitExpr_(const CallNode *op) final {
      bool reduce = IsPtoDirectReduce(GetRef<Call>(op));
      bool fill = IsPtoFill(GetRef<Call>(op));
      if (reduce || fill) {
        scan_->has_direct_reduce |= reduce;
        scan_->has_fill |= fill;
        if (forbidden_depth_ != 0) {
          scan_->has_unsupported_wrapper = true;
          scan_->issue = "direct reduce and fill/clear require a region-level "
                         "statement outside "
                         "Parallel, If/While and thread-bound loops";
        }
      }
      StmtExprVisitor::VisitExpr_(op);
    }

  private:
    PtoRegionScan *scan_;
    int forbidden_depth_ = 0;
  };
  TileOpScanner tileop_scanner(&scan);
  tileop_scanner(region);

  if (scan.has_any_parallel && scan.units.empty()) {
    // A Parallel exists but only under an unsupported wrapper.
    scan.has_unsupported_wrapper = true;
    if (scan.issue.empty()) {
      scan.issue = "T.Parallel exists only under an unsupported wrapper";
    }
  }
  return scan;
}

std::string LaneUseToString(LaneUse use) {
  switch (use) {
  case LaneUse::kUniform:
    return "uniform";
  case LaneUse::kVarying:
    return "varying";
  default:
    return "unknown";
  }
}

namespace {

// Shared per-node classification used by both the value mode and the
// metadata mode: value mode whitelists Add/Mul/Cast; metadata
// mode additionally propagates the pure scalar operators and accepts
// StringImm.
LaneUse AnalyzeLaneUseImpl(const PrimExpr &expr, LaneUseContext *ctx,
                           std::string *reason, bool metadata_mode) {
  if (expr.as<IntImmNode>() || expr.as<FloatImmNode>()) {
    return LaneUse::kUniform;
  }
  if (expr.as<StringImmNode>() != nullptr) {
    // String constants are legal metadata (markers); they never
    // appear as computation values (the value whitelist rejects them).
    if (metadata_mode) {
      return LaneUse::kUniform;
    }
    if (reason != nullptr) {
      // Every Unknown carries a concrete reason.
      *reason = "string constants are metadata-only; they cannot appear "
                "in computation values";
    }
    return LaneUse::kUnknown;
  }
  if (const auto *var = expr.as<VarNode>()) {
    if (var == ctx->loop_var.get()) {
      return LaneUse::kVarying;
    }
    if (ctx->serial_var.defined() && var == ctx->serial_var.get()) {
      // The coordinate of the non-vectorized
      // dimension is a uniform scalar at the program point.
      return LaneUse::kUniform;
    }
    if (ctx->bind_env.count(GetRef<Var>(var))) {
      // Use the recorded classification of the bind (SSA identity).
      if (ctx->classified.count(GetRef<Var>(var))) {
        Integer v = ctx->classified[GetRef<Var>(var)];
        return v->value == 0 ? LaneUse::kUniform : LaneUse::kVarying;
      }
      // On-demand analysis with cycle guard + budget.
      if (ctx->visiting.count(var) != 0) {
        if (reason != nullptr) {
          *reason = "bind self/forward reference detected in lane-use "
                    "analysis";
        }
        return LaneUse::kUnknown;
      }
      if (ctx->budget <= 0) {
        if (reason != nullptr) {
          *reason = "lane-use analysis recursion budget exhausted";
        }
        return LaneUse::kUnknown;
      }
      ctx->visiting.insert(var);
      ctx->budget -= 1;
      const PrimExpr &def = ctx->bind_env[GetRef<Var>(var)];
      LaneUse use = AnalyzeLaneUseImpl(def, ctx, reason, metadata_mode);
      ctx->visiting.erase(var);
      if (use == LaneUse::kUnknown) {
        return use;
      }
      // Record only when the definition is fully analyzed; a cycle hit
      // above leaves it unrecorded so later references re-report.
      ctx->classified.Set(GetRef<Var>(var),
                          Integer(use == LaneUse::kUniform ? 0 : 1));
      return use;
    }
    // Only *known* external scalar vars are Uniform; anything
    // else is undefined -> Unknown.
    if (ctx->external_defs.count(var) != 0) {
      // Being in scope is not enough. An external var consumed as
      // a computation value (or as numeric metadata) must be scalar; a
      // vector-typed var has no scalar input contract for this pass and
      // must not be waved through as Uniform. Buffer data pointers are
      // handle-typed (lanes == 1) and keep flowing through the address
      // and Buffer paths.
      if (var->dtype.lanes() != 1) {
        if (reason != nullptr) {
          std::ostringstream oss;
          oss << "variable `" << var->name_hint << "` has vector dtype "
              << var->dtype
              << "; only scalar inputs are supported as computation values";
          *reason = oss.str();
        }
        return LaneUse::kUnknown;
      }
      return LaneUse::kUniform;
    }
    if (reason != nullptr) {
      std::ostringstream oss;
      oss << "variable `" << var->name_hint
          << "` is not defined in this unit or any outer scope";
      // The diagnostic was built but never written back, leaving
      // metadata errors with an empty reason.
      *reason = oss.str();
    }
    return LaneUse::kUnknown;
  }
  if (const auto *load = expr.as<BufferLoadNode>()) {
    try {
      auto access = AnalyzeBufferAccess(
          load->buffer, load->indices, /*is_write=*/false, ctx->loop_var,
          ctx->mapping, ctx->analyzer, ctx->bind_env);
      if (access.pattern == AccessPattern::kLaneUniform) {
        return LaneUse::kUniform;
      }
      return LaneUse::kVarying;
    } catch (const PtoAnalysisError &err) {
      if (reason != nullptr) {
        std::ostringstream oss;
        oss << "nested access to buffer `" << load->buffer->name
            << "`: " << err.what();
        *reason = oss.str();
      }
      return LaneUse::kUnknown;
    }
  }
  // Hardware-backed unary math shares the whitelist's contract (float32
  // tirx.fabs/exp/log/sqrt): recurse into the single operand.
  if (const auto *call = expr.as<CallNode>()) {
    if (auto call_op = call->op.as<Op>()) {
      const std::string &name = call_op.value()->name;
      if (name == "tirx.fabs" || name == "tirx.exp" || name == "tirx.log" ||
          name == "tirx.sqrt") {
        return AnalyzeLaneUseImpl(call->args[0], ctx, reason, metadata_mode);
      }
    }
  }
  // Binary/relational/boolean operators: both value and metadata modes
  // recurse into operands; the *type* checks live in the callers.
  LaneUse lhs, rhs;
  const PrimExpr *a = nullptr;
  const PrimExpr *b = nullptr;
  bool known_op = false;
  if (const auto *add = expr.as<AddNode>()) {
    a = &add->a;
    b = &add->b;
    known_op = true;
  } else if (const auto *sub = expr.as<SubNode>()) {
    a = &sub->a;
    b = &sub->b;
    known_op = true;
  } else if (const auto *mul = expr.as<MulNode>()) {
    a = &mul->a;
    b = &mul->b;
    known_op = true;
  } else if (const auto *div = expr.as<DivNode>()) {
    a = &div->a;
    b = &div->b;
    known_op = true;
  } else if (const auto *mod = expr.as<ModNode>()) {
    a = &mod->a;
    b = &mod->b;
    known_op = metadata_mode;
  } else if (const auto *fdiv = expr.as<FloorDivNode>()) {
    a = &fdiv->a;
    b = &fdiv->b;
    known_op = metadata_mode;
  } else if (const auto *fmod = expr.as<FloorModNode>()) {
    a = &fmod->a;
    b = &fmod->b;
    known_op = metadata_mode;
  } else if (const auto *minn = expr.as<MinNode>()) {
    a = &minn->a;
    b = &minn->b;
    known_op = metadata_mode;
  } else if (const auto *maxx = expr.as<MaxNode>()) {
    a = &maxx->a;
    b = &maxx->b;
    known_op = metadata_mode;
  } else if (const auto *lt = expr.as<LTNode>()) {
    a = &lt->a;
    b = &lt->b;
    known_op = metadata_mode;
  } else if (const auto *len = expr.as<LENode>()) {
    a = &len->a;
    b = &len->b;
    known_op = metadata_mode;
  } else if (const auto *gt = expr.as<GTNode>()) {
    a = &gt->a;
    b = &gt->b;
    known_op = metadata_mode;
  } else if (const auto *ge = expr.as<GENode>()) {
    a = &ge->a;
    b = &ge->b;
    known_op = metadata_mode;
  } else if (const auto *eq = expr.as<EQNode>()) {
    a = &eq->a;
    b = &eq->b;
    known_op = metadata_mode;
  } else if (const auto *ne = expr.as<NENode>()) {
    a = &ne->a;
    b = &ne->b;
    known_op = metadata_mode;
  } else if (const auto *andn = expr.as<AndNode>()) {
    a = &andn->a;
    b = &andn->b;
    known_op = metadata_mode;
  } else if (const auto *orn = expr.as<OrNode>()) {
    a = &orn->a;
    b = &orn->b;
    known_op = metadata_mode;
  }
  if (known_op) {
    lhs = AnalyzeLaneUseImpl(*a, ctx, reason, metadata_mode);
    rhs = AnalyzeLaneUseImpl(*b, ctx, reason, metadata_mode);
    if (lhs == LaneUse::kUnknown || rhs == LaneUse::kUnknown) {
      if (reason != nullptr && reason->empty()) {
        *reason = "unknown operand in binary expression";
      }
      return LaneUse::kUnknown;
    }
    return (lhs == LaneUse::kUniform && rhs == LaneUse::kUniform)
               ? LaneUse::kUniform
               : LaneUse::kVarying;
  }
  if (!metadata_mode) {
    if (const auto *notn = expr.as<NotNode>()) {
      // Not is a value-mode whitelist reject too; keep symmetric.
    }
  } else {
    if (const auto *notn = expr.as<NotNode>()) {
      return AnalyzeLaneUseImpl(notn->a, ctx, reason, metadata_mode);
    }
  }
  if (const auto *cast = expr.as<CastNode>()) {
    return AnalyzeLaneUseImpl(cast->value, ctx, reason, metadata_mode);
  }
  if (reason != nullptr) {
    std::ostringstream oss;
    oss << "unsupported " << (metadata_mode ? "metadata" : "value") << " node `"
        << expr->GetTypeKey() << "` in lane-use analysis";
    *reason = oss.str();
  }
  return LaneUse::kUnknown;
}

} // namespace

LaneUse AnalyzeLaneUse(const PrimExpr &expr, LaneUseContext *ctx,
                       std::string *reason) {
  return AnalyzeLaneUseImpl(expr, ctx, reason, /*metadata_mode=*/false);
}

LaneUse AnalyzeLaneUseMeta(const PrimExpr &expr, LaneUseContext *ctx,
                           std::string *reason) {
  return AnalyzeLaneUseImpl(expr, ctx, reason, /*metadata_mode=*/true);
}

bool IsSupportedVectorUnaryOp(const std::string &tirx_op, DataType dtype) {
  // Per the VMI ISA documentation: vabs admits f16/bf16/f32 (the vabs
  // instruction also takes si8/si16/si32, but the frontend abs() lowers
  // integers to Select — PR-2 territory — and returns unsigned inputs
  // unchanged, so integer dtypes never reach this gate); vexp/vln/vsqrt
  // admit f16/f32 only.
  const int bits = dtype.bits();
  if (tirx_op == "tirx.fabs") {
    return (dtype.is_float() || dtype.is_bfloat16()) &&
           (bits == 16 || bits == 32);
  }
  if (tirx_op == "tirx.exp" || tirx_op == "tirx.log" ||
      tirx_op == "tirx.sqrt") {
    return dtype.is_float() && (bits == 16 || bits == 32);
  }
  return false;
}

bool ExprUsesVar(const PrimExpr &expr, const VarNode *var) {
  bool uses = false;
  PostOrderVisit(expr, [&](const ObjectRef &n) {
    if (n.as<VarNode>() == var) {
      uses = true;
    }
  });
  return uses;
}

bool ContainsRuntimeValue(const PrimExpr &expr) {
  // True when the expression reads any runtime value: a Var (external
  // scalar, serial coordinate, scalar buffer base) or a BufferLoad.
  // A pure IntImm/FloatImm tree is a compile-time constant.
  bool has_runtime = false;
  PostOrderVisit(expr, [&](const ObjectRef &n) {
    if (n.as<VarNode>() != nullptr || n.as<BufferLoadNode>() != nullptr) {
      has_runtime = true;
    }
  });
  return has_runtime;
}

bool IsFloatFamily(DataType dtype) {
  return dtype.is_float() || dtype.is_bfloat16();
}

bool IsSupportedVectorBinaryOp(const std::string &vmi_op, DataType dtype) {
  // Operation-level matrix (per the VMI ISA documentation):
  //   vmul requires i16/i32/f16/bf16/f32; vadd/vsub additionally allow i8;
  //   vdiv admits f16/f32 only (no bf16, no integers).
  int bits = dtype.bits();
  const bool is_int_like = dtype.is_int() || dtype.is_uint();
  const bool is_fp_like =
      (dtype.is_float() || dtype.is_bfloat16()) && (bits == 16 || bits == 32);
  if (vmi_op == "vmul") {
    return (is_int_like && (bits == 16 || bits == 32)) || is_fp_like;
  }
  if (vmi_op == "vadd" || vmi_op == "vsub") {
    return (is_int_like && (bits == 8 || bits == 16 || bits == 32)) ||
           is_fp_like;
  }
  if (vmi_op == "vdiv") {
    return dtype.is_float() && (bits == 16 || bits == 32);
  }
  return false;
}

namespace {

/*! Cast attributes read from a Cast node's backend annotations. The
 * frontend writes them in two shapes — TIR Imm objects (the `T.cast` helper
 * lowers `round`/`sat` to StringImm/IntImm) and plain Python values (a raw
 * `annotations={...}` map) — so both spellings are accepted here. */
struct CastAttributes {
  /*! The `round` annotation was present (even when its value is empty). */
  bool has_round_key = false;
  /*! A non-empty rounding mode was requested. An empty string is the
   * frontend's way of saying "use the backend default", which is not the same
   * as an unusable mode. */
  bool has_round_mode = false;
  std::string round; // raw spelling, e.g. "rn", "R", "A"
  bool has_sat = false;
  bool sat = true;
  bool has_rbits = false;
};

/*! Read the attribute hints of a Cast. Returns false with *why set when an
 * annotation uses a shape this version cannot read; an unknown key is
 * itself a rejection, because silently ignoring an attribute would accept
 * a conversion whose semantics were never checked. */
bool ReadCastAttributes(const ffi::Map<ffi::String, ffi::Any> &annotations,
                        CastAttributes *attrs, std::string *why) {
  for (const auto &[key_ref, value] : annotations) {
    const std::string key = key_ref;
    if (key == "round") {
      attrs->has_round_key = true;
      if (const auto *imm = value.as<StringImmNode>()) {
        attrs->round = imm->value;
      } else if (auto str = value.as<ffi::String>()) {
        attrs->round = std::string(*str);
      } else {
        *why = "annotation `round` must be a string";
        return false;
      }
      attrs->has_round_mode = !attrs->round.empty();
    } else if (key == "sat") {
      attrs->has_sat = true;
      if (const auto *imm = value.as<IntImmNode>()) {
        // Only 0/1 are accepted, in the TIR Imm form as well.
        if (imm->value != 0 && imm->value != 1) {
          *why = "annotation `sat` must be a boolean (got " +
                 std::to_string(imm->value) + ")";
          return false;
        }
        attrs->sat = imm->value != 0;
      } else if (auto flag = value.as<bool>()) {
        attrs->sat = *flag;
      } else if (auto flag = value.as<int64_t>()) {
        // Only 0/1 are accepted: any other integer is a malformed flag, not a
        // saturating request.
        if (*flag != 0 && *flag != 1) {
          *why = "annotation `sat` must be a boolean (got " +
                 std::to_string(*flag) + ")";
          return false;
        }
        attrs->sat = *flag != 0;
      } else {
        *why = "annotation `sat` must be a boolean";
        return false;
      }
    } else if (key == "rbits") {
      attrs->has_rbits = true;
    } else {
      *why = "unknown cast annotation `" + key + "`";
      return false;
    }
  }
  return true;
}

/*! Normalize a rounding spelling to the canonical PTO token. The frontend
 * aliases are PTX-style; the canonical tokens are the ones the FP8
 * conversion policy reasons about. Returns false with *why set for
 * aliases with no PTO meaning (stochastic rounding) and for tokens no
 * conversion in this version accepts. */
bool NormalizeCastRounding(const std::string &raw, std::string *token,
                           std::string *why) {
  if (raw == "rn") {
    *token = "R";
    return true;
  }
  if (raw == "rz") {
    *token = "Z";
    return true;
  }
  if (raw == "rp") {
    *token = "C";
    return true;
  }
  if (raw == "rm") {
    *token = "F";
    return true;
  }
  if (raw == "rs") {
    *why = "stochastic rounding (round=\"rs\") is not supported";
    return false;
  }
  static const std::set<std::string> kCanonical = {"R", "A", "H",
                                                   "Z", "C", "F"};
  if (kCanonical.count(raw) != 0) {
    *token = raw;
    return true;
  }
  *why = "rounding mode \"" + raw + "\" has no PTO vcvt meaning";
  return false;
}

/*! Rounding tokens the FP8 narrowing policy accepts. The conversion
 * instructions the FP8 target exposes are FP32 <-> FP8; the matrix is the
 * intersection of the PTODSL vcvt interface with that support, so the source
 * is always float32 and only the rounding modes below are expressible. */
const std::set<std::string> &Fp8AllowedRoundings() {
  static const std::set<std::string> kAllowed = {"R", "A", "H", "Z"};
  return kAllowed;
}

bool IsVmiLogicalLaneCount(int64_t lanes) {
  return lanes == 64 || lanes == 128 || lanes == 256;
}

std::string CastContextName(const CastContext &context) {
  if (context.value_kind == CastValueKind::kScalar) {
    return "scalar";
  }
  std::ostringstream oss;
  oss << "VMI, lanes=" << context.logical_lanes;
  return oss.str();
}

bool IsHalfOrBfloat16(DataType dtype) {
  return (dtype.is_float() && dtype.bits() == 16) || dtype.is_bfloat16();
}

/*! Identity attribute policy shared by the scalar and VMI branches: no
 * conversion exists, so sat=True (the default) is normalized away, while
 * sat=False or a non-empty rounding mode would claim a conversion semantic
 * that does not exist.  An explicit empty round key means "no request". */
bool ValidateIdentityCastAttributes(const CastAttributes &attrs,
                                    const std::string &pair, std::string *why) {
  if (attrs.has_sat && !attrs.sat) {
    *why = pair + ": an identity cast does not accept sat=False";
    return false;
  }
  if (attrs.has_round_mode) {
    *why = pair + ": an identity cast does not accept a rounding mode";
    return false;
  }
  return true;
}

/*! Integer attribute policy: rounding modes are never accepted; an explicit
 * saturation flag is accepted only for narrowing and only as sat=False, which
 * the direct NOSAT lowering already honours. */
bool ValidateIntegerCastAttributes(const CastAttributes &attrs, bool narrowing,
                                   const std::string &pair, std::string *why) {
  if (attrs.has_round_mode) {
    *why = pair + ": integer conversions do not accept a rounding mode";
    return false;
  }
  if (attrs.has_sat) {
    if (!narrowing) {
      *why = pair + ": this integer conversion does not accept an explicit "
                    "saturation flag";
      return false;
    }
    if (attrs.sat) {
      *why = pair + ": integer narrowing accepts only sat=False (NOSAT), not "
                    "sat=True";
      return false;
    }
  }
  return true;
}

Optional<CastPlan> PlanScalarValueCast(const CastNode *cast,
                                       const CastAttributes &attrs,
                                       const std::string &pair,
                                       std::string *reason) {
  const DataType from = cast->value.dtype();
  const DataType to = cast->dtype;
  auto reject = [&](const std::string &why) {
    if (reason != nullptr) {
      *reason = why;
    }
    return Optional<CastPlan>();
  };
  ScalarCastPlan keep;
  keep.action = ScalarCastAction::kKeepInput;
  ScalarCastPlan rebuild;
  rebuild.action = ScalarCastAction::kRebuildCast;
  if (from == to) {
    std::string why;
    if (!ValidateIdentityCastAttributes(attrs, pair, &why)) {
      return reject(why);
    }
    return CastPlan{keep};
  }
  const bool from_int = from.is_int() || from.is_uint();
  const bool to_int = to.is_int() || to.is_uint();
  if (from_int && to_int) {
    std::string why;
    if (!ValidateIntegerCastAttributes(attrs, to.bits() < from.bits(), pair,
                                       &why)) {
      return reject(why);
    }
    // Scalar integer casts keep the ordinary pto.cast/trunc lowering; the
    // validated sat=False marker is already honoured and is normalized away.
    return CastPlan{rebuild};
  }
  if (IsSupportedFloat8(from) || IsSupportedFloat8(to)) {
    return reject(pair + ": non-identity FP8 conversions have no scalar "
                         "lowering inside SIMD_VF; use a lane-varying value "
                         "so the cast takes the VMI vcvt path");
  }
  const bool from_float = from.is_float() || from.is_bfloat16();
  const bool to_float = to.is_float() || to.is_bfloat16();
  if (from_float && to_float) {
    if (attrs.has_round_mode) {
      return reject(pair + ": this conversion does not accept a rounding mode");
    }
    const bool from_f32 = from.is_float() && from.bits() == 32;
    const bool to_f32 = to.is_float() && to.bits() == 32;
    if (!from_f32 && to_f32) {
      if (attrs.has_sat) {
        return reject(pair + ": this direction does not accept a saturation "
                             "mode");
      }
      return CastPlan{rebuild};
    }
    if (from_f32 && !to_f32) {
      if (attrs.has_sat && attrs.sat) {
        return reject(pair + ": this conversion accepts only sat=False "
                             "(non-saturating)");
      }
      return CastPlan{rebuild};
    }
    return reject(pair + ": cross-family f16 <-> bf16 narrowing has no "
                         "validated rounding semantics; convert through "
                         "float32 explicitly");
  }
  return reject(pair + ": no scalar conversion exists for this dtype pair");
}

Optional<CastPlan>
PlanVectorValueCast(const CastNode *cast, const CastContext &context,
                    const CastAttributes &attrs, const std::string &round_token,
                    const std::string &pair, std::string *reason) {
  const DataType from = cast->value.dtype();
  const DataType to = cast->dtype;
  auto reject = [&](const std::string &why) {
    if (reason != nullptr) {
      *reason = why;
    }
    return Optional<CastPlan>();
  };
  if (from == to) {
    std::string why;
    if (!ValidateIdentityCastAttributes(attrs, pair, &why)) {
      return reject(why);
    }
    return CastPlan{VectorCastPlan{}};
  }
  const bool from_int = from.is_int() || from.is_uint();
  const bool to_int = to.is_int() || to.is_uint();
  if (from_int && to_int) {
    std::string why;
    if (!ValidateIntegerCastAttributes(attrs, to.bits() < from.bits(), pair,
                                       &why)) {
      return reject(why);
    }
    VectorCastPlan plan;
    if (to.bits() < from.bits()) {
      plan.steps.push_back(
          {VectorCastOp::kIntegerTruncate, from, to, "", "NOSAT"});
      return CastPlan{plan};
    }
    if (to.bits() == from.bits()) {
      plan.steps.push_back(
          {VectorCastOp::kIntegerReinterpret, from, to, "", ""});
      return CastPlan{plan};
    }
    if (from.is_int() == to.is_int()) {
      plan.steps.push_back({VectorCastOp::kIntegerExtend, from, to, "", ""});
      return CastPlan{plan};
    }
    // Cross-sign widening: extend with the source signedness to the target
    // width, then re-view the same-width result as the target signedness.
    const DataType intermediate =
        from.is_int() ? DataType::Int(to.bits()) : DataType::UInt(to.bits());
    plan.steps.push_back(
        {VectorCastOp::kIntegerExtend, from, intermediate, "", ""});
    plan.steps.push_back(
        {VectorCastOp::kIntegerReinterpret, intermediate, to, "", ""});
    return CastPlan{plan};
  }
  const bool from_fp8 = IsSupportedFloat8(from);
  const bool to_fp8 = IsSupportedFloat8(to);
  if (from_fp8 || to_fp8) {
    if (from_fp8 && to_fp8) {
      return reject(pair + ": conversions between the two FP8 types are not "
                           "supported (they are not a reinterpretation)");
    }
    VectorCastPlan plan;
    if (from_fp8 && to.is_float() && to.bits() == 32) {
      // Exact widening: a real conversion with no options.  An explicit hint
      // here cannot be honoured, so it is rejected rather than dropped.
      if (attrs.has_round_key) {
        return reject(pair + ": this direction does not accept a rounding "
                             "mode");
      }
      if (attrs.has_sat) {
        return reject(pair + ": this direction does not accept a saturation "
                             "mode");
      }
      plan.steps.push_back(
          {VectorCastOp::kNonIntegerConvert, from, to, "", ""});
      return CastPlan{plan};
    }
    if (to_fp8) {
      if (!(from.is_float() && from.bits() == 32)) {
        return reject(pair + ": the FP8 conversion instructions take a "
                             "float32 source; convert to float32 explicitly "
                             "first");
      }
      const std::set<std::string> &allowed = Fp8AllowedRoundings();
      const std::string token = attrs.has_round_mode ? round_token : "R";
      if (allowed.count(token) == 0) {
        std::ostringstream oss;
        oss << pair << ": rounding mode " << token;
        if (attrs.has_round_mode && attrs.round != token) {
          oss << " (from round=\"" << attrs.round << "\")";
        }
        oss << " is not supported by the FP8 conversion policy";
        return reject(oss.str());
      }
      plan.steps.push_back({VectorCastOp::kNonIntegerConvert, from, to, token,
                            (attrs.has_sat && !attrs.sat) ? "NOSAT" : "SAT"});
      return CastPlan{plan};
    }
    return reject(pair + ": FP8 -> FP16/BF16 and FP8 <-> integer are out of "
                         "scope; convert through float32 explicitly");
  }
  const bool from_float = from.is_float() || from.is_bfloat16();
  const bool to_float = to.is_float() || to.is_bfloat16();
  if (from_float && to_float) {
    if (attrs.has_round_mode) {
      return reject(pair + ": this conversion does not accept a rounding mode");
    }
    const bool from_f32 = from.is_float() && from.bits() == 32;
    const bool to_f32 = to.is_float() && to.bits() == 32;
    VectorCastPlan plan;
    if (!from_f32 && to_f32) {
      if (attrs.has_sat) {
        return reject(pair + ": this direction does not accept a saturation "
                             "mode");
      }
      // A real conversion: empty options are not a no-op.
      plan.steps.push_back(
          {VectorCastOp::kNonIntegerConvert, from, to, "", ""});
      return CastPlan{plan};
    }
    if (from_f32 && !to_f32) {
      if (attrs.has_sat && attrs.sat) {
        return reject(pair + ": this conversion accepts only sat=False "
                             "(NOSAT)");
      }
      // Pin R/NOSAT so the generated vcvt never depends on PTODSL defaults.
      plan.steps.push_back(
          {VectorCastOp::kNonIntegerConvert, from, to, "R", "NOSAT"});
      return CastPlan{plan};
    }
    return reject(pair + ": cross-family f16 <-> bf16 narrowing has no "
                         "validated rounding semantics; convert through "
                         "float32 explicitly");
  }
  return reject(pair + ": float <-> integer conversions need per-combination "
                       "rounding/overflow validation");
}

} // namespace

Optional<CastPlan> PlanValueCast(const CastNode *cast,
                                 const CastContext &context,
                                 std::string *reason) {
  DataType from = cast->value.dtype();
  DataType to = cast->dtype;
  auto reject = [&](const std::string &why) {
    if (reason != nullptr) {
      *reason = why;
    }
    return Optional<CastPlan>();
  };
  std::ostringstream pair_os;
  pair_os << "`" << from << " -> " << to << "` (" << CastContextName(context)
          << ")";
  const std::string pair = pair_os.str();
  // The explicit kind is authoritative; lanes only validate it.  Inferring
  // the kind from lanes would turn a lane-1 VMI value into a scalar and a
  // scalar request with vector lanes into an implicit VMI fallback.
  if (context.value_kind == CastValueKind::kScalar) {
    if (context.logical_lanes != 1) {
      return reject(pair +
                    ": a scalar cast context must have exactly one "
                    "logical lane, got " +
                    std::to_string(context.logical_lanes));
    }
  } else {
    if (!IsVmiLogicalLaneCount(context.logical_lanes)) {
      return reject(pair +
                    ": a VMI cast context requires logical lanes "
                    "64/128/256, got " +
                    std::to_string(context.logical_lanes));
    }
  }
  // Dtype legality is a precondition for every action plan,
  // including identity — unsupported types (int24/FP64/...) must not slip
  // through the same-type fast path.  The Parallel value path is
  // element-wise, so both sides must be lane-1 elements.
  if (from.lanes() != 1 || to.lanes() != 1) {
    return reject(pair + ": value casts require lane-1 element operands");
  }
  if (!IsSupportedElementDType(from) || !IsSupportedElementDType(to)) {
    return reject(pair + " is outside the supported element dtype set");
  }

  // Attributes are parsed for every Cast, not just the FP8 ones: the
  // expression whitelist deliberately does not look inside a cast's
  // annotations (attributes are not computation values), so an annotation the
  // selected conversion cannot honour has to be rejected here — for every
  // direction — or it would be dropped silently.
  CastAttributes attrs;
  std::string why;
  if (!ReadCastAttributes(cast->annotations, &attrs, &why)) {
    return reject(pair + ": " + why);
  }
  if (attrs.has_rbits) {
    return reject(pair + ": the rbits operand (stochastic rounding) is not "
                         "supported");
  }
  std::string round_token;
  if (attrs.has_round_mode &&
      !NormalizeCastRounding(attrs.round, &round_token, &why)) {
    return reject(pair + ": " + why);
  }

  if (context.value_kind == CastValueKind::kScalar) {
    return PlanScalarValueCast(cast, attrs, pair, reason);
  }
  return PlanVectorValueCast(cast, context, attrs, round_token, pair, reason);
}

std::optional<std::string> FindUnsupportedExprNode(const PrimExpr &expr) {
  WhitelistVisitor visitor;
  visitor(expr);
  return visitor.unsupported;
}

} // namespace pto
} // namespace tl
} // namespace tvm
