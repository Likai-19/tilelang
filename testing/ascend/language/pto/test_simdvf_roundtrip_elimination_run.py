"""Device numeric tests for EliminateSimdVFRoundTrip (cf5v / cf2v shapes).

cf5v is the shape whose UB round trip made the result flaky on device: a
``T.serial`` scalar loop computes ``quot``, then a SimdVF region copies ``a``
into a region-local buffer, compares ``|a| < eps`` and copies ``y`` back out.
After the pass the vecscope no longer routes the value through UB, so it cannot
be read back stale.

Two protocols run here:

* a plain golden comparison (the pass must not change the result at all);
* the polluter protocol: a kernel first paints the whole UB window with 777.0,
  then the measured kernel runs. UB content survives across kernel launches on
  device, so any read of an unlanded value shows up deterministically (this is
  what turned the pre-fix flake into a 20/20 failure).

Assertions are bitwise via :func:`numpy.isclose` with the same tolerances the
acceptance probe uses (rtol=1e-5, atol=1e-6); "mostly right" is a failure.
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

E, K, LANES = 256, 8, 64
EPS = 1e-4
POLLUTE_N = 8192
NEG = float(-3.402823e38)
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


def _build_cf5v(eps=EPS):
    """quot (serial) + |a| < eps ? 0 : quot, through SimdVF round-trip buffers."""

    @T.prim_func
    def main(A: T.Tensor((E,), "float32"), B: T.Tensor((E,), "float32"),
             Y: T.Tensor((E,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((E,), "float32")
            b_ub = T.alloc_shared((E,), "float32")
            y_ub = T.alloc_shared((E,), "float32")
            quot = T.alloc_shared((E,), "float32")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            for i in T.serial(E):
                quot[i] = a_ub[i] / b_ub[i]
            with T.SimdVF(lanes=LANES):
                a = T.alloc_shared((E,), "float32")
                y = T.alloc_shared((E,), "float32")
                for i in T.Parallel(E):
                    a[i] = a_ub[i]
                for i in T.Parallel(E):
                    ax = T.Select(a[i] >= T.float32(0.0), a[i],
                                  a[i] * T.float32(-1.0))
                    y[i] = T.Select(ax < eps, T.float32(0.0), quot[i])
                for i in T.Parallel(E):
                    y_ub[i] = y[i]
            T.copy(y_ub, Y)

    return main


def _build_cf2v():
    """The loop-carried shape the pass must leave alone."""

    @T.prim_func
    def main(A: T.Tensor((E,), "float32"), Thresh: T.Tensor((K,), "float32"),
             Out: T.Tensor((E,), "float32")):
        with T.Kernel(1):
            s = T.alloc_shared((E,), "float32")
            thr_ub = T.alloc_shared((K,), "float32")
            out_ub = T.alloc_shared((E,), "float32")
            T.copy(A, s)
            T.copy(Thresh, thr_ub)
            with T.SimdVF(lanes=LANES):
                scores = T.alloc_shared((E,), "float32")
                for k in T.serial(K):
                    thr = thr_ub[k]
                    for i in T.Parallel(E):
                        scores[i] = s[i]
                    for i in T.Parallel(E):
                        s[i] = T.Select(scores[i] > thr, T.float32(NEG), s[i])
                for i in T.Parallel(E):
                    out_ub[i] = s[i]
            T.copy(out_ub, Out)

    return main


# ---------------------------------------------------------------------------
# goldens
# ---------------------------------------------------------------------------


def _gold_cf5(a, b, eps=EPS):
    q = (a / b).astype(np.float32)
    ax = np.abs(a).astype(np.float32)
    return np.where(ax < np.float32(eps), np.float32(0.0), q).astype(np.float32)


def _gold_cf2(a, thr):
    s = a.astype(np.float32).copy()
    for k in range(len(thr)):
        scores = s.copy()
        s = np.where(scores > np.float32(thr[k]), np.float32(NEG), s).astype(np.float32)
    return s


def _compile(prim):
    return tilelang.compile(prim, target="pto", out_idx=-1)


def _run_and_compare(kernel, args, gold, tag=""):
    got = kernel(*args).cpu().numpy()
    bad = int((~np.isclose(got, gold, rtol=RTOL, atol=ATOL)).sum())
    assert bad == 0, f"{tag}: {bad}/{len(gold)} mismatched; first: {np.flatnonzero(~np.isclose(got, gold, rtol=RTOL, atol=ATOL))[:5]}"
    return got


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


@pytest.mark.pto
@requires_npu
def test_cf5v_numeric():
    rng = np.random.default_rng(0)
    a = rng.uniform(-1.0, 1.0, E).astype(np.float32)
    b = rng.uniform(0.5, 2.0, E).astype(np.float32)
    a[:8] = 0.0  # forces the |a| < eps arm on the zeroed lanes
    gold = _gold_cf5(a, b)

    kernel = _compile(_build_cf5v())
    _run_and_compare(
        kernel,
        (torch.from_numpy(a).to(_dev()), torch.from_numpy(b).to(_dev())),
        gold,
        "cf5v",
    )


@pytest.mark.pto
@requires_npu
def test_cf5v_source_has_no_roundtrip():
    kernel = _compile(_build_cf5v())
    source = kernel.get_kernel_source()
    assert source.count("pto.vmi.vstore(") == 1, "expected one vstore (y_ub only)"
    assert source.count("for _tl_range_q in range(0, 4, 1)") == 1
    # The S -> V edge that publishes `quot` must survive.
    assert 'pto.wait_flag("S", "V", event_id=0)' in source


@pytest.mark.pto
@requires_npu
def test_cf5v_after_ub_pollution():
    """The pre-fix failure mode: run behind a kernel that paints UB with 777.0."""
    rng = np.random.default_rng(1)
    a = rng.uniform(-1.0, 1.0, E).astype(np.float32)
    b = rng.uniform(0.5, 2.0, E).astype(np.float32)
    a[:8] = 0.0
    gold = _gold_cf5(a, b)

    polluter = _compile(_build_polluter())
    kernel = _compile(_build_cf5v())
    args = (torch.from_numpy(a).to(_dev()), torch.from_numpy(b).to(_dev()))

    rounds = int(os.environ.get("ROUNDS", "5"))
    for r in range(rounds):
        polluter()
        torch.npu.synchronize()
        _run_and_compare(kernel, args, gold, f"cf5v polluter round {r}")


@pytest.mark.pto
@requires_npu
def test_cf2v_numeric():
    a = np.linspace(-3.0, 5.0, E).astype(np.float32)
    thr = np.linspace(0.0, 4.0, K).astype(np.float32)
    gold = _gold_cf2(a, thr)

    kernel = _compile(_build_cf2v())
    _run_and_compare(
        kernel,
        (torch.from_numpy(a).to(_dev()), torch.from_numpy(thr).to(_dev())),
        gold,
        "cf2v",
    )
