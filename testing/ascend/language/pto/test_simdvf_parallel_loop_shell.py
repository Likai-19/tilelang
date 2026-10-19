"""Loop-shell tests: constant-bound T.serial inside T.Parallel (Form B).

Hardware independent: each case lowers TileLang IR for the PTO target with
:func:`tilelang.engine.lower.lower` and inspects the module produced by
``VectorizeParallelToPTO`` (no PTOAS/device compile unless the case is
explicitly numeric).

Expected contract after this stage:

* a constant-bound ``T.serial`` loop inside ``T.Parallel`` is admitted by
  Verify as a preserved shell and copied through as a scalar loop; only
  its body is vectorized;
* when the canonical carry pattern matches (a seed store immediately
  before the shell, then same-index self read-modify-write inside it),
  the carried value is promoted to a ``local.var`` vreg buffer — the
  seed store and the in-shell stores are eliminated and one writeback
  ``vstore`` is emitted after the shell (the UB round-trip form is not
  order-safe on the current PTOAS: no automatic mem_bar);
* a shell whose carry cannot be promoted is rejected with a
  ``[VectorizeParallelToPTO] loop shell`` diagnostic instead of being
  silently mis-lowered;
* a shell without a memory-carried value (no buffer both read and
  written inside the loop) is preserved without promotion.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

import tilelang.ascend.language as T
from tilelang.ascend import transform as ascend_transform
from tilelang.ascend.target import normalize_pto_target
from tilelang.engine.lower import lower as _lower
import tilelang
import tvm


@contextmanager
def _stage_capture():
    captured = {}
    original_legalize = ascend_transform.LegalizeParallelToPTO
    original_vectorize = ascend_transform.VectorizeParallelToPTO

    def capturing_legalize_factory():
        legalize_pass = original_legalize()

        def run(mod):
            result = legalize_pass(mod)
            captured["legalize"] = result
            return result

        return run

    def capturing_vectorize_factory():
        vectorize_pass = original_vectorize()

        def run(mod):
            result = vectorize_pass(mod)
            captured["vectorize"] = result
            return result

        return run

    ascend_transform.LegalizeParallelToPTO = capturing_legalize_factory
    ascend_transform.VectorizeParallelToPTO = capturing_vectorize_factory
    try:
        yield captured
    finally:
        ascend_transform.LegalizeParallelToPTO = original_legalize
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


def _serial_shells_named(mod, name):
    found = []

    def visit(node):
        if isinstance(node, tvm.tirx.For) and node.loop_var.name == name:
            found.append(node)

    tvm.tirx.stmt_functor.post_order_visit(_body_of(mod), visit)
    return found


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

def _k_newton_inside(E, lanes, dtype="float32"):
    """Seed store + 3-step Newton self-RMW on y_ub: the canonical carry."""
    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(X: T.Tensor((P,), dtype), Y: T.Tensor((P,), dtype)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((P,), dtype)
            y_ub = T.alloc_shared((P,), dtype)
            T.copy(X, x_ub)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    y_ub[i] = T.float32(1.0)
                    for _t in T.serial(3):
                        y_ub[i] = y_ub[i] * (T.float32(2.0) - x_ub[i] * y_ub[i])
            T.copy(y_ub, Y)

    return main


def _k_shell_no_carry(E, lanes, dtype="float32"):
    """Loop body only reads x and writes out: no carried value, no seed."""
    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(X: T.Tensor((P,), dtype), Y: T.Tensor((P,), dtype)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((P,), dtype)
            out_ub = T.alloc_shared((P,), dtype)
            T.copy(X, x_ub)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    for _t in T.serial(3):
                        out_ub[i] = x_ub[i] * T.float32(2.0)
            T.copy(out_ub, Y)

    return main


def _k_two_carried(E=256, lanes=64):
    """Two buffers both read and written inside the shell: rejected."""
    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(X: T.Tensor((P,), "float32"), Y: T.Tensor((P,), "float32")):
        with T.Kernel(1):
            x_ub = T.alloc_shared((P,), "float32")
            y_ub = T.alloc_shared((P,), "float32")
            T.copy(X, x_ub)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    y_ub[i] = T.float32(1.0)
                    x_ub[i] = T.float32(0.5)
                    for _t in T.serial(3):
                        y_ub[i] = y_ub[i] + x_ub[i]
                        x_ub[i] = y_ub[i] * T.float32(2.0)
            T.copy(y_ub, Y)

    return main


def _k_seed_not_adjacent(E=256, lanes=64):
    """Seed store exists but is not the statement right before the shell."""
    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(X: T.Tensor((P,), "float32"), Y: T.Tensor((P,), "float32")):
        with T.Kernel(1):
            x_ub = T.alloc_shared((P,), "float32")
            y_ub = T.alloc_shared((P,), "float32")
            z_ub = T.alloc_shared((P,), "float32")
            T.copy(X, x_ub)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    y_ub[i] = T.float32(1.0)
                    z_ub[i] = T.float32(0.0)
                    for _t in T.serial(3):
                        y_ub[i] = y_ub[i] * (T.float32(2.0) - x_ub[i] * y_ub[i])
            T.copy(y_ub, Y)

    return main


def _k_region_leak(E=256, lanes=64):
    """A second unit reads the carried buffer: region isolation violated."""
    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(X: T.Tensor((P,), "float32"), Y: T.Tensor((P,), "float32")):
        with T.Kernel(1):
            x_ub = T.alloc_shared((P,), "float32")
            y_ub = T.alloc_shared((P,), "float32")
            out_ub = T.alloc_shared((P,), "float32")
            T.copy(X, x_ub)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    y_ub[i] = T.float32(1.0)
                    for _t in T.serial(3):
                        y_ub[i] = y_ub[i] * (T.float32(2.0) - x_ub[i] * y_ub[i])
                for i in T.Parallel(E):
                    out_ub[i] = y_ub[i]
            T.copy(out_ub, Y)

    return main


# ---------------------------------------------------------------------------
# Positives
# ---------------------------------------------------------------------------

@pytest.mark.pto
def test_inside_serial_promotes_carry_to_vreg():
    E, lanes = 256, 64
    stages, error = _lower_capture(_k_newton_inside(E, lanes))
    assert error is None, f"the inside-serial form must lower, got: {error}"

    vec = str(stages["vectorize"])
    # The carried value lives in a local.var vreg buffer, not in UB.
    assert "y_ub_carry" in vec, "the carry must be promoted to a vreg buffer"
    assert "local.var" in vec
    # Seed store and in-shell stores are eliminated; exactly one vstore
    # (the post-shell writeback) remains.
    assert vec.count("vstore") == 1, (
        f"expected exactly one writeback vstore, got {vec.count('vstore')}"
    )
    # The shell is preserved as a scalar loop over _t.
    shells = _serial_shells_named(stages["vectorize"], "_t")
    assert shells, "the serial shell must survive vectorization"

    # Legalize is identity for loops: still scalar, no VMI.
    legal = str(stages["legalize"])
    assert "vload" not in legal and "vstore" not in legal


@pytest.mark.pto
def test_inside_serial_tail_mask_with_carry():
    E, lanes = 150, 128  # non-divisible: dynamic tail mask + carry
    stages, error = _lower_capture(_k_newton_inside(E, lanes))
    assert error is None, f"the tail + carry form must lower, got: {error}"

    vec = str(stages["vectorize"])
    assert "y_ub_carry" in vec
    assert "remaining" in vec, "the tail path must keep its remaining counter"
    assert vec.count("vstore") == 1
    assert _serial_shells_named(stages["vectorize"], "_t")


@pytest.mark.pto
def test_shell_without_carry_is_preserved():
    E, lanes = 256, 64
    stages, error = _lower_capture(_k_shell_no_carry(E, lanes))
    assert error is None, f"the carry-free shell must lower, got: {error}"

    vec = str(stages["vectorize"])
    assert "_carry" not in vec, "no carry pattern: nothing must be promoted"
    assert vec.count("vstore") == 1, "the single out store stays a vstore"
    assert _serial_shells_named(stages["vectorize"], "_t")


# ---------------------------------------------------------------------------
# Negatives: order-unsafe shells are rejected, not silently mis-lowered
# ---------------------------------------------------------------------------

@pytest.mark.pto
def test_two_carried_buffers_are_rejected():
    _, error = _lower_capture(_k_two_carried(), expect_error=True)
    assert error is not None
    message = str(error)
    assert "[VectorizeParallelToPTO] loop shell" in message
    assert "more than one buffer" in message


@pytest.mark.pto
def test_seed_not_adjacent_is_rejected():
    _, error = _lower_capture(_k_seed_not_adjacent(), expect_error=True)
    assert error is not None
    message = str(error)
    assert "[VectorizeParallelToPTO] loop shell" in message
    assert "seed" in message


@pytest.mark.pto
def test_carried_buffer_read_elsewhere_is_rejected():
    _, error = _lower_capture(_k_region_leak(), expect_error=True)
    assert error is not None
    message = str(error)
    assert "[VectorizeParallelToPTO] loop shell" in message
    assert "only be accessed" in message
