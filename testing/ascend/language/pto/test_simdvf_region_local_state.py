"""Region-local state inside a SimdVF region.

Two facts that share the same region-level state handling:

* a per-lane ``T.alloc_var`` becomes a ``local.var`` vreg, one vreg per unit
  and never shared between units;
* a UB allocation written and read across a vecscope split is hoisted to a
  single allocation site in the parent block.

Both are structure-only assertions: the pass output is inspected directly.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

import tilelang.ascend.language as T
from tilelang.ascend import transform as ascend_transform
from tilelang.ascend.target import normalize_pto_target
from tilelang.engine.lower import lower as _lower
import tvm


@contextmanager
def _stage_capture():
    captured = {}
    original_vectorize = ascend_transform.VectorizeParallelToPTO

    def capturing_vectorize_factory():
        vectorize_pass = original_vectorize()

        def run(mod):
            result = vectorize_pass(mod)
            captured["vectorize"] = result
            return result

        return run

    ascend_transform.VectorizeParallelToPTO = capturing_vectorize_factory
    try:
        yield captured
    finally:
        ascend_transform.VectorizeParallelToPTO = original_vectorize


def _lower_capture(func, *, expect_error=False):
    target = normalize_pto_target("pto")
    error = None
    with _stage_capture() as captured:
        try:
            with target:
                _lower(func, target=target)
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            error = exc
    if error is not None and not expect_error:
        raise error
    return captured, error


def _body_of(mod):
    assert len(mod.functions) == 1, f"expected a single function, got {list(mod.functions)}"
    return next(iter(mod.functions.values())).body


def _accessed_buffers(mod):
    """name -> Buffer for every buffer read or written in the vectorized body."""
    found = {}

    def visit(node):
        if isinstance(node, (tvm.tirx.BufferStore, tvm.tirx.BufferLoad)):
            found[node.buffer.name] = node.buffer

    tvm.tirx.stmt_functor.post_order_visit(_body_of(mod), visit)
    return found


def _stmt_text(mod):
    return tvm.ir.save_json(_body_of(mod))


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

def _k_absmax_accumulator(E=256, lanes=64, dtype="float32"):
    """Scalar absmax accumulator over a constant-bound shell (sv2v shape)."""
    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(X: T.Tensor((P,), dtype), Y: T.Tensor((P,), dtype)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((P,), dtype)
            y_ub = T.alloc_shared((P,), dtype)
            T.copy(X, x_ub)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    m = T.alloc_var(dtype, init=0.0)
                    for _c in T.serial(3):
                        _v = x_ub[(i + _c * lanes) % P]
                        m = T.max(m, T.max(_v, -_v))
                    y_ub[i] = T.max(m, T.float32(1e-6))
            T.copy(y_ub, Y)

    return main


def _k_lane_varying_seed(E=256, lanes=64, dtype="float32"):
    """Scalar seeded from a lane-varying load, then used as a divisor."""
    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(S: T.Tensor((P,), dtype), X: T.Tensor((P,), dtype),
             Y: T.Tensor((P,), dtype)):
        with T.Kernel(1):
            s_ub = T.alloc_shared((P,), dtype)
            x_ub = T.alloc_shared((P,), dtype)
            y_ub = T.alloc_shared((P,), dtype)
            T.copy(S, s_ub)
            T.copy(X, x_ub)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    d = T.alloc_var(dtype)
                    d = s_ub[i]
                    y_ub[i] = x_ub[i] / d
            T.copy(y_ub, Y)

    return main


def _k_region_level_alloc_var(E=256, lanes=64, dtype="float32"):
    """alloc_var used outside the Parallel unit: no promotion path."""
    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(X: T.Tensor((P,), dtype), Y: T.Tensor((P,), dtype)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((P,), dtype)
            y_ub = T.alloc_shared((P,), dtype)
            T.copy(X, x_ub)
            with T.SimdVF(lanes=lanes):
                acc = T.alloc_var(dtype, init=0.0)
                for i in T.Parallel(E):
                    y_ub[i] = x_ub[i] + T.Cast(dtype, acc)
            T.copy(y_ub, Y)

    return main


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "builder,source_name",
    [
        (_k_absmax_accumulator, "m"),
        (_k_lane_varying_seed, "d"),
    ],
)
def test_lane_scalar_becomes_a_local_var_vreg(builder, source_name):
    """A per-lane scalar -- accumulator or lane-varying seed -- is promoted to
    a ``local.var`` vreg and the source buffer disappears from the accesses."""
    captured, _ = _lower_capture(builder())
    buffers = _accessed_buffers(captured["vectorize"])
    lane = {n: b for n, b in buffers.items() if n.endswith("_lane")}
    assert lane, f"no promoted vreg in {sorted(buffers)}"
    (name, buf), = lane.items()
    assert buf.scope() == "local.var"
    assert buf.dtype.lanes == 64, buf.dtype
    assert source_name not in buffers, sorted(buffers)
    # The final value is read back from the vreg.
    assert name in _stmt_text(captured["vectorize"])


def test_promotion_is_stable_across_two_units():
    """Two units each own their scalar: two distinct vregs, no sharing."""
    captured, _ = _lower_capture(_k_absmax_accumulator())
    text = _stmt_text(captured["vectorize"])
    assert "m_lane" in text


def test_region_level_alloc_var_rejected():
    _, error = _lower_capture(_k_region_level_alloc_var(), expect_error=True)
    assert error is not None
    message = str(error)
    assert "local.var" in message or "lane-uniform" in message, message


def _alloc_sites(mod, name):
    """Every SBlock that declares `name` in alloc_buffers."""
    sites = []

    def visit(node):
        if isinstance(node, tvm.tirx.SBlock):
            for buf in node.alloc_buffers:
                if buf.name == name:
                    sites.append(node)

    tvm.tirx.stmt_functor.post_order_visit(_body_of(mod), visit)
    return sites


def _access_count(mod, name):
    n = [0]

    def visit(node):
        if isinstance(node, (tvm.tirx.BufferStore, tvm.tirx.BufferLoad)):
            if node.buffer.name == name:
                n[0] += 1

    tvm.tirx.stmt_functor.post_order_visit(_body_of(mod), visit)
    return n[0]


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

def _k_region_local_carrier(E=256, K=8, lanes=64):
    """cf1v shape: region-local `scores` carried through a serial loop."""
    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(A: T.Tensor((P,), "float32"), Thr: T.Tensor((K,), "float32"),
             Out: T.Tensor((P,), "float32")):
        with T.Kernel(1):
            s_ub = T.alloc_shared((P,), "float32")
            thr_ub = T.alloc_shared((K,), "float32")
            out_ub = T.alloc_shared((P,), "float32")
            T.copy(A, s_ub)
            T.copy(Thr, thr_ub)
            with T.SimdVF(lanes=lanes):
                scores = T.alloc_shared((E,), "float32")
                for i in T.Parallel(E):
                    scores[i] = s_ub[i]
                for k in T.serial(K):
                    thr = thr_ub[k]
                    for i in T.Parallel(E):
                        scores[i] = T.Select(scores[i] > thr, T.float32(-3.402823e38), scores[i])
                for i in T.Parallel(E):
                    out_ub[i] = scores[i]
            T.copy(out_ub, Out)

    return main


def _k_neighbour_region_local(E=256, lanes=64):
    """B.3 shape: `tmp` is not the carrier but lives across the split."""
    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(A: T.Tensor((P,), "float32"), Out: T.Tensor((P,), "float32")):
        with T.Kernel(1):
            s_ub = T.alloc_shared((P,), "float32")
            out_ub = T.alloc_shared((P,), "float32")
            T.copy(A, s_ub)
            with T.SimdVF(lanes=lanes):
                scores = T.alloc_shared((E,), "float32")
                tmp = T.alloc_shared((E,), "float32")
                for i in T.Parallel(E):
                    scores[i] = s_ub[i]
                for i in T.Parallel(E):
                    tmp[i] = s_ub[i] * T.float32(2.0)
                for k in T.serial(4):
                    for i in T.Parallel(E):
                        scores[i] = T.Select(scores[i] > T.float32(0.0),
                                             T.float32(-3.402823e38), scores[i])
                for i in T.Parallel(E):
                    out_ub[i] = scores[i] + tmp[i]
            T.copy(out_ub, Out)

    return main


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_region_local_carrier_has_single_allocation():
    captured, _ = _lower_capture(_k_region_local_carrier())
    mod = captured["vectorize"]
    sites = _alloc_sites(mod, "scores")
    assert len(sites) == 1, f"expected one allocation site for `scores`, got {len(sites)}"
    # Still accessed by all three parts.
    assert _access_count(mod, "scores") >= 4


def test_neighbour_region_local_is_hoisted_too():
    captured, _ = _lower_capture(_k_neighbour_region_local())
    mod = captured["vectorize"]
    for name in ("scores", "tmp"):
        sites = _alloc_sites(mod, name)
        assert len(sites) == 1, f"expected one allocation site for `{name}`, got {len(sites)}"
    assert _access_count(mod, "tmp") >= 2


def test_no_split_keeps_region_local_allocation():
    """A region with no carried loop is converted as a whole: no hoisting."""
    P = 256

    @T.prim_func
    def main(A: T.Tensor((P,), "float32"), Out: T.Tensor((P,), "float32")):
        with T.Kernel(1):
            s_ub = T.alloc_shared((P,), "float32")
            out_ub = T.alloc_shared((P,), "float32")
            T.copy(A, s_ub)
            with T.SimdVF(lanes=64):
                t = T.alloc_shared((P,), "float32")
                for i in T.Parallel(P):
                    t[i] = s_ub[i] * T.float32(2.0)
                for i in T.Parallel(P):
                    out_ub[i] = t[i] + T.float32(1.0)
            T.copy(out_ub, Out)

    captured, _ = _lower_capture(main)
    sites = _alloc_sites(captured["vectorize"], "t")
    assert len(sites) == 1
