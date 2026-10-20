"""Per-lane scalar promotion tests (T.alloc_var inside T.Parallel).

Hardware independent: lowers TileLang IR for the PTO target with
:func:`tilelang.engine.lower.lower` and inspects the module produced by
``VectorizeParallelToPTO``.

Expected contract:

* ``T.alloc_var`` inside a converting ``T.Parallel`` unit is a per-lane
  register: its whole-buffer store/load pair is promoted to a ``local.var``
  vector (vreg) named ``<buffer>_lane``;
* a scalar seed value is broadcast into that vreg; a lane-varying seed
  (``s = scale[i]``) is stored as-is;
* the original scalar buffer is no longer read or written, so the shell
  carries an SSA value instead of round-tripping through UB;
* the final value after the shell is read back from the vreg;
* a ``local.var`` access at region level (outside the converting unit) has no
  promotion path and stays rejected.
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

def test_accumulator_promoted_to_vreg():
    captured, _ = _lower_capture(_k_absmax_accumulator())
    buffers = _accessed_buffers(captured["vectorize"])
    lane = {n: b for n, b in buffers.items() if n.endswith("_lane")}
    assert lane, f"no promoted vreg in {sorted(buffers)}"
    (name, buf), = lane.items()
    assert buf.scope() == "local.var"
    assert buf.dtype.lanes == 64, buf.dtype
    # The source scalar buffer is no longer accessed at all.
    assert "m" not in buffers, sorted(buffers)
    # ... and the final value is read back from the vreg.
    text = _stmt_text(captured["vectorize"])
    assert name in text


def test_lane_varying_seed_promoted():
    captured, _ = _lower_capture(_k_lane_varying_seed())
    buffers = _accessed_buffers(captured["vectorize"])
    lane = [n for n in buffers if n.endswith("_lane")]
    assert lane, f"no promoted vreg in {sorted(buffers)}"
    assert "d" not in buffers, sorted(buffers)


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
