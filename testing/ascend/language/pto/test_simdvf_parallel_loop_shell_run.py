"""Device numeric tests for runtime trip counts in a preserved loop shell.

A ``T.serial`` loop inside ``T.Parallel`` is copied through as a scalar shell,
so its trip count does not have to be known at compile time. These tests pin
the runtime behaviour, including the two boundaries that only show up when the
count is dynamic:

* ``iters == 0``: the shell body never runs, so the carried value stays at the
  seed value (the writeback must not invent a value);
* ``iters > 0``: the in-register carry accumulates every iteration exactly
  once — no double counting from the prologue/epilogue around the shell.
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


def _build_carried_runtime_shell():
    """seed 0 + runtime-bound accumulate: ``y = iters * x``."""

    @T.prim_func
    def main(X: T.Tensor((E,), "float32"), Y: T.Tensor((E,), "float32"),
             iters: T.int32):
        with T.Kernel(1):
            x_ub = T.alloc_shared((E,), "float32")
            y_ub = T.alloc_shared((E,), "float32")
            T.copy(X, x_ub)
            with T.SimdVF(lanes=LANES):
                for i in T.Parallel(E):
                    y_ub[i] = T.float32(0.0)
                    for _t in T.serial(iters):
                        y_ub[i] = y_ub[i] + x_ub[i]
            T.copy(y_ub, Y)

    return main


def _build_plain_runtime_shell():
    """no carry: the loop only writes ``y`` with ``2 * x`` (never reads it)."""

    @T.prim_func
    def main(X: T.Tensor((E,), "float32"), Y: T.Tensor((E,), "float32"),
             iters: T.int32):
        with T.Kernel(1):
            x_ub = T.alloc_shared((E,), "float32")
            y_ub = T.alloc_shared((E,), "float32")
            T.copy(X, x_ub)
            with T.SimdVF(lanes=LANES):
                # Seed the buffer first: with ``iters == 0`` the output must be
                # this seed (the loop body is not a carry, so nothing else can
                # produce a value).
                for i in T.Parallel(E):
                    y_ub[i] = T.float32(-7.0)
                for i in T.Parallel(E):
                    for _t in T.serial(iters):
                        y_ub[i] = x_ub[i] * T.float32(2.0)
            T.copy(y_ub, Y)

    return main


@pytest.mark.pto
@requires_npu
@pytest.mark.parametrize("iters", [0, 1, 4])
def test_carried_runtime_shell_counts_every_iteration(iters):
    rng = np.random.default_rng(11)
    x = rng.uniform(-2.0, 2.0, E).astype(np.float32)
    # The kernel takes a scalar argument, so the output cannot be derived from
    # ``out_idx`` (a scalar parameter is not a tensor): pass the buffer in.
    kernel = tilelang.compile(_build_carried_runtime_shell(), target="pto")
    out = torch.empty(E, dtype=torch.float32, device=_dev())
    kernel(torch.from_numpy(x).to(_dev()), out, iters)
    got = out.cpu().numpy().astype(np.float32)
    want = (x * np.float32(iters)).astype(np.float32)
    bad = int((~np.isclose(got, want, rtol=1e-5, atol=1e-6)).sum())
    assert bad == 0, f"iters={iters}: {bad}/{E} mismatched"


@pytest.mark.pto
@requires_npu
@pytest.mark.parametrize("iters", [0, 3])
def test_plain_runtime_shell_runs_or_skips(iters):
    rng = np.random.default_rng(12)
    x = rng.uniform(-2.0, 2.0, E).astype(np.float32)
    kernel = tilelang.compile(_build_plain_runtime_shell(), target="pto")
    out = torch.full((E,), -7.0, dtype=torch.float32, device=_dev())
    kernel(torch.from_numpy(x).to(_dev()), out, iters)
    got = out.cpu().numpy().astype(np.float32)
    if iters == 0:
        # The shell never runs, so the store inside it never executes and the
        # caller's buffer keeps its contents: pin that (no phantom write).
        assert np.all(got == np.float32(-7.0)), (
            f"iters=0 must not write, got {got[:4]}"
        )
        return
    want = (x * np.float32(2.0)).astype(np.float32)
    bad = int((~np.isclose(got, want, rtol=1e-5, atol=1e-6)).sum())
    assert bad == 0, f"iters={iters}: {bad}/{E} mismatched"
