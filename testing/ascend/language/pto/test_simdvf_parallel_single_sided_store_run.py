"""Device numeric tests for one-sided conditional writes in a converting
region.

A one-sided branch (``if cond: B[i] = v``) may only write the lanes the
condition selects. Lowering it as a Select over a read of ``B`` would add a
UB read the source program does not have; since a vecscope has no ordering
point, that read observes whatever UB held before — including a value written
by an earlier unit of the same region. The store therefore has to be
predicated; these tests pin the numeric result, including behind a UB
polluter, which makes any stale read deterministic instead of incidental.
"""

import os
import sys

import numpy as np
import pytest
import torch

try:  # the CPU wheel has no `torch.npu`; the device image registers it here
    import torch_npu  # noqa: F401
except ImportError:
    pass

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ["TILELANG_DISABLE_DATA_RACE_CHECK"] = "1"

import tilelang
import tilelang.ascend.language as T

E, LANES = 256, 64
K = 8
POLLUTE_N = 8192
RTOL, ATOL = 1e-5, 1e-6


def _npu_available():
    if not hasattr(torch, "npu"):
        return False
    try:
        torch.randn(1, device=torch.device("npu"))
        torch.npu.synchronize()
        return True
    except Exception:  # noqa: BLE001
        return False


_NPU = _npu_available()
requires_npu = pytest.mark.skipif(not _NPU, reason="NPU runtime is unavailable")


def _dev():
    return torch.device("npu")


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


def _build_polluter(n=POLLUTE_N):
    """Paint the UB window with 777.0 and copy it out."""

    @T.prim_func
    def main(Out: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            buf = T.alloc_shared((n,), "float32")
            with T.SimdVF(lanes=64):
                for i in T.Parallel(n):
                    buf[i] = T.float32(777.0)
            T.copy(buf, Out)

    return main


def _build_single_sided():
    """Unit 1 writes every lane, unit 2 overwrites one side of the lane set."""

    @T.prim_func
    def main(X: T.Tensor((E,), "float32"), Y: T.Tensor((E,), "float32")):
        with T.Kernel(1):
            x_ub = T.alloc_shared((E,), "float32")
            y_ub = T.alloc_shared((E,), "float32")
            T.copy(X, x_ub)
            with T.SimdVF(lanes=LANES):
                for i in T.Parallel(E):
                    y_ub[i] = T.float32(7.0)
                for i in T.Parallel(E):
                    if x_ub[i] > T.float32(0.0):
                        y_ub[i] = T.float32(1.0)
            T.copy(y_ub, Y)

    return main


def _build_nested_single_sided():
    @T.prim_func
    def main(X: T.Tensor((E,), "float32"), Z: T.Tensor((E,), "float32"),
             Y: T.Tensor((E,), "float32")):
        with T.Kernel(1):
            x_ub = T.alloc_shared((E,), "float32")
            z_ub = T.alloc_shared((E,), "float32")
            y_ub = T.alloc_shared((E,), "float32")
            T.copy(X, x_ub)
            T.copy(Z, z_ub)
            with T.SimdVF(lanes=LANES):
                for i in T.Parallel(E):
                    y_ub[i] = T.float32(7.0)
                for i in T.Parallel(E):
                    if x_ub[i] > T.float32(0.0):
                        if z_ub[i] > T.float32(0.0):
                            y_ub[i] = T.float32(1.0)
            T.copy(y_ub, Y)

    return main


def _build_nested_then_else():
    """One-sided inner branch inside a then-arm that also has an else-arm.

    The inner predicate must survive the outer rewrite: lanes selected by the
    outer condition but not by the inner one keep the previous value.
    """

    @T.prim_func
    def main(A: T.Tensor((E,), "float32"), B: T.Tensor((E,), "float32"),
             Y: T.Tensor((E,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((E,), "float32")
            b_ub = T.alloc_shared((E,), "float32")
            y_ub = T.alloc_shared((E,), "float32")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF(lanes=LANES):
                for i in T.Parallel(E):
                    y_ub[i] = T.float32(7.0)
                for i in T.Parallel(E):
                    if a_ub[i] > T.float32(0.0):
                        if b_ub[i] > T.float32(0.0):
                            y_ub[i] = T.float32(1.0)
                    else:
                        y_ub[i] = T.float32(2.0)
            T.copy(y_ub, Y)

    return main


def _build_or_condition():
    @T.prim_func
    def main(X: T.Tensor((E,), "float32"), Z: T.Tensor((E,), "float32"),
             Y: T.Tensor((E,), "float32")):
        with T.Kernel(1):
            x_ub = T.alloc_shared((E,), "float32")
            z_ub = T.alloc_shared((E,), "float32")
            y_ub = T.alloc_shared((E,), "float32")
            T.copy(X, x_ub)
            T.copy(Z, z_ub)
            with T.SimdVF(lanes=LANES):
                for i in T.Parallel(E):
                    y_ub[i] = T.float32(7.0)
                for i in T.Parallel(E):
                    if (x_ub[i] > T.float32(0.0)) or (z_ub[i] > T.float32(0.0)):
                        y_ub[i] = T.float32(1.0)
            T.copy(y_ub, Y)

    return main


def _build_carried_conditional():
    """One-sided write to the loop-carried buffer: the update keeps the
    register value on the lanes the condition does not select."""

    @T.prim_func
    def main(X: T.Tensor((E,), "float32"), Thresh: T.Tensor((K,), "float32"),
             Y: T.Tensor((E,), "float32")):
        with T.Kernel(1):
            x_ub = T.alloc_shared((E,), "float32")
            thr_ub = T.alloc_shared((K,), "float32")
            y_ub = T.alloc_shared((E,), "float32")
            T.copy(X, x_ub)
            T.copy(Thresh, thr_ub)
            with T.SimdVF(lanes=LANES):
                acc = T.alloc_shared((E,), "float32")
                for i in T.Parallel(E):
                    acc[i] = T.float32(0.0)
                for k in T.serial(K):
                    thr = thr_ub[k]
                    for i in T.Parallel(E):
                        if x_ub[i] > thr:
                            acc[i] = acc[i] + T.float32(1.0)
                for i in T.Parallel(E):
                    y_ub[i] = acc[i]
            T.copy(y_ub, Y)

    return main


# ---------------------------------------------------------------------------
# goldens
# ---------------------------------------------------------------------------


def _gold_single_sided(x):
    return np.where(x > 0, np.float32(1.0), np.float32(7.0)).astype(np.float32)


def _gold_nested(x, z):
    return np.where((x > 0) & (z > 0), np.float32(1.0), np.float32(7.0)).astype(np.float32)


def _gold_nested_then_else(a, b):
    return np.where(a > 0, np.where(b > 0, np.float32(1.0), np.float32(7.0)),
                    np.float32(2.0)).astype(np.float32)


def _gold_or(x, z):
    return np.where((x > 0) | (z > 0), np.float32(1.0), np.float32(7.0)).astype(np.float32)


def _gold_carried(x, thr):
    acc = np.zeros_like(x)
    for k in range(len(thr)):
        acc = np.where(x > np.float32(thr[k]), acc + np.float32(1.0), acc).astype(np.float32)
    return acc


def _compile(prim):
    return tilelang.compile(prim, target="pto", out_idx=-1)


def _run_and_compare(kernel, args, gold, tag=""):
    got = kernel(*args).cpu().numpy()
    bad = int((~np.isclose(got, gold, rtol=RTOL, atol=ATOL)).sum())
    assert bad == 0, (
        f"{tag}: {bad}/{len(gold)} mismatched; "
        f"first: {np.flatnonzero(~np.isclose(got, gold, rtol=RTOL, atol=ATOL))[:5]}"
    )
    return got


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


@pytest.mark.pto
@requires_npu
def test_single_sided_store_numeric():
    """Numeric, then again behind a 777.0 UB polluter: a read of the
    destination would surface the polluting value on the lanes the condition
    does not select."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-2.0, 2.0, E).astype(np.float32)
    gold = _gold_single_sided(x)
    args = (torch.from_numpy(x).to(_dev()),)

    _run_and_compare(_compile(_build_single_sided()), args, gold, "single-sided")

    polluter = _compile(_build_polluter())
    kernel = _compile(_build_single_sided())
    for r in range(int(os.environ.get("ROUNDS", "3"))):
        polluter()
        torch.npu.synchronize()
        _run_and_compare(kernel, args, gold, f"single-sided polluter round {r}")


@pytest.mark.pto
@requires_npu
@pytest.mark.parametrize(
    "builder,gold,tag",
    [
        (_build_nested_single_sided, lambda x, z: _gold_nested(x, z), "nested single-sided"),
        (_build_or_condition, lambda x, z: _gold_or(x, z), "or-condition single-sided"),
    ],
)
def test_compound_conditions_numeric(builder, gold, tag):
    rng = np.random.default_rng(2)
    x = rng.uniform(-2.0, 2.0, E).astype(np.float32)
    z = rng.uniform(-2.0, 2.0, E).astype(np.float32)
    kernel = _compile(builder())
    _run_and_compare(
        kernel,
        (torch.from_numpy(x).to(_dev()), torch.from_numpy(z).to(_dev())),
        gold(x, z),
        tag,
    )


@pytest.mark.pto
@requires_npu
def test_nested_then_else_keeps_the_inner_predicate():
    """The inner one-sided branch must not be widened when the outer branch
    pairs with an else-arm."""
    rng = np.random.default_rng(4)
    a = rng.uniform(-2.0, 2.0, E).astype(np.float32)
    b = rng.uniform(-2.0, 2.0, E).astype(np.float32)
    kernel = _compile(_build_nested_then_else())
    _run_and_compare(
        kernel,
        (torch.from_numpy(a).to(_dev()), torch.from_numpy(b).to(_dev())),
        _gold_nested_then_else(a, b),
        "nested then/else",
    )


@pytest.mark.pto
@requires_npu
def test_carried_conditional_update_numeric():
    """A one-sided update of the carried buffer must keep the register value
    on the unselected lanes and stay ordered across iterations."""
    x = np.linspace(-2.0, 2.0, E).astype(np.float32)
    thr = np.linspace(-1.5, 1.5, K).astype(np.float32)
    kernel = _compile(_build_carried_conditional())
    _run_and_compare(
        kernel,
        (torch.from_numpy(x).to(_dev()), torch.from_numpy(thr).to(_dev())),
        _gold_carried(x, thr),
        "carried conditional",
    )


@pytest.mark.pto
@requires_npu
def test_single_sided_store_source_is_predicated():
    """The destination must not be read back: no vload of it, and the store
    carries a computed mask."""
    kernel = _compile(_build_single_sided())
    source = kernel.get_kernel_source()
    # The predicate is conjoined into the store mask, not used as the store's
    # only mask: the second vstore must consume a bound `mand_N` value.
    vstores = [ln for ln in source.splitlines() if "pto.vmi.vstore(" in ln]
    assert len(vstores) == 2, f"expected two vstores, got {len(vstores)}"
    assert "mand_" in vstores[1], (
        f"the conditional store must consume a conjoined mask, got: {vstores[1].strip()[:120]}"
    )
    # The destination is never read: only the source buffer is loaded.
    assert source.count("pto.vmi.vload(") == 1, (
        "only the source buffer may be loaded, got "
        f"{source.count('pto.vmi.vload(')} vloads"
    )
