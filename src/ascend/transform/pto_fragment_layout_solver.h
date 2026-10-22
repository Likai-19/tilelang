/*!
 * \file pto_fragment_layout_solver.h
 * \brief Parity-based joint layout solver for PTO fragment buffers
 * (R4 sub-design, rewritten).
 *
 * Runs once per function, after LayoutInference's inference-object
 * collection and before the engine's staged inference:
 *
 *  1. Pre-scan (all classified rejections, no diagnostics deferral):
 *     fragment accesses outside converting T.Parallel bodies (A2),
 *     fragment Buffer views sharing one data Var, SimdVF-scope layout
 *     annotations, non-canonical kernel-level annotations, and residual
 *     fragment-touching tile operators inside converting regions.
 *  2. Per converting SIMD_VF region: build the parity graph — a binary
 *     direction variable for every 2D Parallel unit (which loop dim is
 *     vectorized) and a binary lane-dim variable for every 2D fragment
 *     (which buffer dim maps to lanes). UB accesses give unary
 *     constraints on the direction (the continuity proof only holds for
 *     shared scopes); every (unit, fragment) access pair gives a partial
 *     function direction -> lane-dim.
 *  3. Solve with a parity union-find: unary constraints propagate
 *     first, every connected component is either contradictory (a
 *     proven conflict, reported with the originating units and the two
 *     candidate layouts) or has exactly two mutually flipped solutions.
 *  4. Pick per component: the smaller total backing size wins; ties are
 *     broken by the source-order-first unit with a free direction
 *     taking the inner (loop) dimension. Unconstrained fragments take
 *     the byte-minimal lane dim (ties: the last dim, matching the inner
 *     preference).
 *  5. Emit one canonical Fragment layout per fragment buffer of every
 *     converting region — including fragments that are never accessed
 *     (floating), so the engine's step 0 cannot seed them as
 *     FullyReplicated (B5).
 *
 * The layouts are injected into the engine's initial layout map as
 * pre-determined facts; InferPtoSimdLayout skips re-proposing injected
 * entries and its dimension selection consumes them as direction
 * constraints (fragments exit the UB continuity proof — B1).
 */

#ifndef TVM_TL_ASCEND_TRANSFORM_PTO_FRAGMENT_LAYOUT_SOLVER_H_
#define TVM_TL_ASCEND_TRANSFORM_PTO_FRAGMENT_LAYOUT_SOLVER_H_

#include <tvm/ir/expr.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>

#include <optional>
#include <unordered_map>

#include "../../layout/layout.h"
#include "../../op/operator.h"

namespace tvm {
namespace tl {
namespace pto {

/*! \brief Solve the fragment layouts of every converting PTO SIMD_VF
 * region in \p body.
 *
 * \param body the function body (post collection, pre inference).
 * \param target_is_pto the collector already resolved the target; the
 *        solver is a no-op (nullopt) for non-PTO functions.
 * \param annotated_layouts layout-map annotations already loaded by the
 *        engine (kernel-level T.annotate_layout). Entries targeting a
 *        region fragment are validated against the canonical forms and
 *        act as pinned lane-dim constraints; SimdVF-scope and
 *        non-canonical annotations are rejected here.
 * \return the solved canonical layout per fragment buffer, or nullopt
 *         when the function has no converting region.
 * \throws Error (classified, user-facing) for every pre-scan rejection
 *         and for proven parity conflicts. */
std::optional<LayoutMap>
SolvePtoFragmentLayouts(const tirx::Stmt &body, bool target_is_pto,
                        const LayoutMap &annotated_layouts);

/*! \brief The lane dimension of a canonical PTO fragment layout: the
 * input dim the forward_thread expression depends on. Returns -1 when
 * the layout is not of the canonical single-floor-mod form. */
int PtoFragmentLaneDim(const Fragment &layout);

}  // namespace pto
}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_ASCEND_TRANSFORM_PTO_FRAGMENT_LAYOUT_SOLVER_H_
