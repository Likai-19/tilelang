"""End-to-end NPU tests for the fragment-backing (PR-4 slot storage) path.

These kernels keep a ``T.alloc_fragment`` buffer across sequential
``T.Parallel`` units inside ``T.SimdVF``: VectorizeParallelToPTO converts the
fragment accesses into ordinary ``vload``/``vstore`` on an exclusively
reserved unified-buffer backing (entry q at element offset q*lanes, chunk
mask on stores) and orders each backing-writing unit with a vecscope
``mem_bar("VST_VLD")``. The mem-to-value promotion (PTOAS MR 274) may then
forward the store->load round trips into registers; without it the round
trips execute through the backing and the mem_bar keeps them ordered, so
numerics must hold on either toolchain.
"""

import os
import sys

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ["TILELANG_DISABLE_DATA_RACE_CHECK"] = "1"

import tilelang
import tilelang.ascend.language as T


def _require_npu_runtime():
    if not hasattr(torch, "npu"):
        pytest.skip("torch.npu is unavailable in the current environment")

    try:
        _ = torch.randn(1, device=torch.device("npu"))
        torch.npu.synchronize()
    except Exception as err:
        pytest.skip(f"NPU runtime is unavailable in the current environment: {err}")


@pytest.fixture()
def npu():
    _require_npu_runtime()
    return torch.device("npu")


def _run_and_check(func, inputs, expected, source_checks=None, atol=None):
    kernel = tilelang.compile(func, target="pto", out_idx=-1)
    source = kernel.get_kernel_source()
    if source_checks is not None:
        source_checks(source)
    out = kernel(*inputs)
    torch.npu.synchronize()
    got = out.cpu()
    if atol is None:
        assert torch.equal(got, expected), f"numeric mismatch: max_diff={torch.max(torch.abs((got.float() - expected.float()))).item()}"
    else:
        max_diff = torch.max(torch.abs(got.float() - expected.float())).item()
        assert torch.allclose(got.float(), expected.float(), rtol=0, atol=atol), f"numeric mismatch: max_diff={max_diff}"


def _assert_backing_source():
    """Ordering and shape checks on the emitted PTODSL. The backing buffer
    itself merges into buf_dyn_shmem (its name does not survive); here we
    verify the declaration form, the vecscope and the mem_bar fences. The
    mem-to-value promotion may forward some round trips and drop their
    bars; on the toolchains these kernels are validated against the
    backing-writing bars are observed to survive, and this assert pins
    that."""

    def check(source):
        assert "pto.memref_declare(" in source, (
            "fragment storage must be declared through pto.memref_declare"
        )
        assert "pto.mem_bar" in source, "backing-writing units must be ordered by mem_bar"
        assert "with pto.vecscope():" in source

    return check


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def _sv3_keep(CM, lanes, K):
    """GEMV partial-sum keep: acc[CM, VL] lives across a serial K loop
    (zero-init -> K accumulation rounds -> write-back), the S1 acceptance
    ladder kernel."""

    @T.prim_func
    def main(
        A: T.Buffer((K, CM * lanes), "float32"),
        X: T.Buffer((K,), "float32"),
        Out: T.Buffer((CM * lanes,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((K, CM * lanes), "float32")
            x_ub = T.alloc_shared((K,), "float32")
            out_ub = T.alloc_shared((CM * lanes,), "float32")
            T.copy(A, a_ub)
            T.copy(X, x_ub)
            with T.SimdVF(lanes=lanes):
                acc = T.alloc_fragment((CM, lanes), "float32")
                for i, j in T.Parallel(CM, lanes):
                    acc[i, j] = T.float32(0.0)
                for k in T.serial(K):
                    for i, j in T.Parallel(CM, lanes):
                        acc[i, j] = acc[i, j] + a_ub[k, i * lanes + j] * x_ub[k]
                for i, j in T.Parallel(CM, lanes):
                    out_ub[i * lanes + j] = acc[i, j]
            T.copy(out_ub, Out)

    return main


def _conditional_kill(E, lanes):
    """cf1 shape: a 1D fragment kept across units with a single-sided
    conditional store (kills entries above a threshold); the predicate is
    recovered from the legalized Select form and merged into the store mask
    (no old-value read, no vsel on the write-back path)."""

    @T.prim_func
    def main(
        S: T.Buffer((E,), "float32"),
        Thr: T.Buffer((1,), "float32"),
        Out: T.Buffer((E,), "float32"),
    ):
        with T.Kernel(1):
            s_ub = T.alloc_shared((E,), "float32")
            thr_ub = T.alloc_shared((1,), "float32")
            out_ub = T.alloc_shared((E,), "float32")
            T.copy(S, s_ub)
            T.copy(Thr, thr_ub)
            with T.SimdVF(lanes=lanes):
                scores = T.alloc_fragment((E,), "float32")
                for i in T.Parallel(E):
                    scores[i] = s_ub[i]
                for i in T.Parallel(E):
                    if scores[i] > thr_ub[0]:
                        scores[i] = T.float32(-1.0)
                for i in T.Parallel(E):
                    out_ub[i] = scores[i]
            T.copy(out_ub, Out)

    return main


def _keep_ragged_1d(E, lanes):
    """1D fragment over a ragged E (non-divisible): ceil(E/L) entries, the
    tail entry's padding lanes must not leak into the compact output."""

    @T.prim_func
    def main(
        A: T.Buffer((E,), "float32"),
        Out: T.Buffer((E,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((E,), "float32")
            out_ub = T.alloc_shared((E,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                keep = T.alloc_fragment((E,), "float32")
                for i in T.Parallel(E):
                    keep[i] = a_ub[i] * T.float32(2.0)
                for i in T.Parallel(E):
                    out_ub[i] = keep[i] + T.float32(1.0)
            T.copy(out_ub, Out)

    return main


def _fill_then_mask(E, lanes):
    """Fragment initialized by a region-level T.fill, then partially
    rewritten by a Parallel unit: the fill expands into a Parallel store
    round and the later unit only touches the covered lanes, so the fill
    value must survive in the untouched lanes."""

    @T.prim_func
    def main(
        A: T.Buffer((E,), "float32"),
        Out: T.Buffer((E,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((E,), "float32")
            out_ub = T.alloc_shared((E,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                keep = T.alloc_fragment((E,), "float32")
                T.fill(keep, T.float32(7.0))
                for i in T.Parallel(E):
                    if a_ub[i] > T.float32(0.0):
                        keep[i] = a_ub[i]
                for i in T.Parallel(E):
                    out_ub[i] = keep[i]
            T.copy(out_ub, Out)

    return main


def _copy_out_of_fragment(E, lanes):
    """A region-level T.copy out of a fragment: the copy expands into a
    Parallel round whose load goes through the fragment handle and whose
    store writes the UB side through the ordinary continuous path. The
    producer is read-modify-write so the base round-trip elimination cannot
    forward the fragment away before Vectorize."""

    @T.prim_func
    def main(
        A: T.Buffer((E,), "float32"),
        Out: T.Buffer((E,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((E,), "float32")
            out_ub = T.alloc_shared((E,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                keep = T.alloc_fragment((E,), "float32")
                for i in T.Parallel(E):
                    keep[i] = a_ub[i] * T.float32(3.0)
                for i in T.Parallel(E):
                    keep[i] = keep[i] + T.float32(1.0)
                T.copy(keep, out_ub)
            T.copy(out_ub, Out)

    return main


def _multi_chunk_keep(M, N, lanes):
    """A 2D fragment wider than one vector (N a multiple of VL): each row
    spans several entries and the backing offset distributes into
    i*(Q*L)+q*L; the whole-entry granularity must hold and every chunk must
    read back the data its row stored."""

    @T.prim_func
    def main(
        A: T.Buffer((M, N), "float32"),
        Out: T.Buffer((M, N), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((M, N), "float32")
            out_ub = T.alloc_shared((M, N), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                keep = T.alloc_fragment((M, N), "float32")
                for i, j in T.Parallel(M, N):
                    keep[i, j] = a_ub[i, j] * T.float32(2.0)
                for i, j in T.Parallel(M, N):
                    out_ub[i, j] = keep[i, j] + T.float32(1.0)
            T.copy(out_ub, Out)

    return main


def _first_dim_lanes(M, N, lanes):
    """B1 shape: lanes fall on the FIRST logical dimension (the UB access
    is transposed): the joint solver picks lane dim 0 and the whole
    round trip must carry the transposition."""

    @T.prim_func
    def main(
        A: T.Buffer((N, M), "float32"),
        Out: T.Buffer((N, M), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N, M), "float32")
            out_ub = T.alloc_shared((N, M), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                keep = T.alloc_fragment((M, N), "float32")
                for i, j in T.Parallel(M, N):
                    keep[i, j] = a_ub[j, i]
                for i, j in T.Parallel(M, N):
                    out_ub[j, i] = keep[i, j] * T.float32(2.0)
            T.copy(out_ub, Out)

    return main


def _unaligned_fill(E, lanes):
    """G2 shape: an unaligned fill region initializes the middle chunk;
    lanes outside the region keep the fragment's previous value."""

    @T.prim_func
    def main(
        A: T.Buffer((E,), "float32"),
        Out: T.Buffer((E,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((E,), "float32")
            out_ub = T.alloc_shared((E,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                keep = T.alloc_fragment((E,), "float32")
                for i in T.Parallel(E):
                    keep[i] = a_ub[i]
                T.fill(keep[lanes // 2 : lanes // 2 + lanes], T.float32(7.0))
                for i in T.Parallel(E):
                    out_ub[i] = keep[i]
            T.copy(out_ub, Out)

    return main


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.pto
@pytest.mark.parametrize("lanes", [64, 128, 256])
def test_sv3_keep_numeric(npu, lanes):
    CM, K = 24, 5
    a = torch.randn(K, CM * lanes)
    x = torch.randn(K)
    expected = (a * x[:, None]).sum(dim=0)
    _run_and_check(
        _sv3_keep(CM, lanes, K),
        (a.to(npu), x.to(npu)),
        expected,
        source_checks=_assert_backing_source(),
        atol=1e-3,
    )


@pytest.mark.pto
@pytest.mark.parametrize("lanes", [64, 128])
def test_conditional_kill_numeric(npu, lanes):
    E = lanes * 2 + 17  # ragged tail on purpose
    s = torch.randn(E)
    thr = torch.tensor([0.1])
    expected = torch.where(s > thr[0], torch.tensor(-1.0), s)
    _run_and_check(
        _conditional_kill(E, lanes),
        (
            s.to(npu),
            thr.to(npu),
        ),
        expected,
        source_checks=_assert_backing_source(),
    )


@pytest.mark.pto
@pytest.mark.parametrize("lanes", [64, 128, 256])
def test_keep_ragged_1d_numeric(npu, lanes):
    E = lanes + lanes // 2 + 3  # tail entry with padding lanes
    a = torch.randn(E)
    _run_and_check(
        _keep_ragged_1d(E, lanes),
        (a.to(npu),),
        a * 2.0 + 1.0,
        source_checks=_assert_backing_source(),
    )


@pytest.mark.pto
@pytest.mark.parametrize("lanes", [64, 128])
def test_fill_then_mask_numeric(npu, lanes):
    E = lanes * 2 + 17  # ragged tail on purpose
    a = torch.randn(E)
    expected = torch.where(a > 0.0, a, torch.tensor(7.0))
    _run_and_check(
        _fill_then_mask(E, lanes),
        (a.to(npu),),
        expected,
        source_checks=_assert_backing_source(),
    )


@pytest.mark.pto
@pytest.mark.parametrize("lanes", [64, 128, 256])
def test_copy_out_of_fragment_numeric(npu, lanes):
    E = lanes + lanes // 2 + 3  # ragged tail on purpose
    a = torch.randn(E)
    _run_and_check(
        _copy_out_of_fragment(E, lanes),
        (a.to(npu),),
        a * 3.0 + 1.0,
        source_checks=_assert_backing_source(),
    )


@pytest.mark.pto
@pytest.mark.parametrize("lanes", [64, 128, 256])
def test_multi_chunk_keep_numeric(npu, lanes):
    M = 4
    N = 2 * lanes  # two entries per row
    a = torch.randn(M, N)
    _run_and_check(
        _multi_chunk_keep(M, N, lanes),
        (a.to(npu),),
        a * 2.0 + 1.0,
        source_checks=_assert_backing_source(),
    )


@pytest.mark.pto
@pytest.mark.parametrize("lanes", [128])
def test_first_dim_lanes_numeric(npu, lanes):
    M, N = 256, 4
    a = torch.randn(N, M)
    _run_and_check(
        _first_dim_lanes(M, N, lanes),
        (a.to(npu),),
        a * 2.0,
        source_checks=_assert_backing_source(),
    )


@pytest.mark.pto
@pytest.mark.parametrize("lanes", [128])
def test_unaligned_fill_numeric(npu, lanes):
    E = 2 * lanes
    a = torch.randn(E)
    expected = a.clone()
    expected[lanes // 2 : lanes // 2 + lanes] = 7.0
    _run_and_check(
        _unaligned_fill(E, lanes),
        (a.to(npu),),
        expected,
        source_checks=_assert_backing_source(),
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
