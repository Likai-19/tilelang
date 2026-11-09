"""Positive NPU correctness coverage for high-level PTO direct reductions.

Every case calls T.reduce_* and tilelang.compile(target="pto")
without replacement hooks. Source/destination arguments cover their complete
logical Buffer views. Explicit allocated owners include initialized vector-read
padding and destination canaries; larger owners are not passed as sliced
reduction regions. Automatic scheduling remains enabled.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.ascend.language as T


@pytest.fixture(scope="module")
def npu():
    assert hasattr(torch, "npu") and torch.npu.is_available(), "NPU is required for direct-reduce correctness"
    device = torch.device("npu")
    source = torch.arange(8, dtype=torch.float32)
    torch.testing.assert_close(source.to(device).cpu(), source)
    return device


TORCH_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "uint32": torch.uint32,
}
ELEMENT_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2, "int32": 4, "uint32": 4}
OPS = ("sum", "max", "min")
ABS_OPS = ("abssum", "absmax")


def _ceil(value, unit):
    return (value + unit - 1) // unit * unit


def _storage(shape, lanes, dtype, transpose=False, compact=False):
    """Return real strides/capacity with initialized per-vector read padding."""
    width = max(lanes, 256 // ELEMENT_BYTES[dtype])
    if len(shape) == 1:
        return (1,), _ceil(shape[0], width) + width
    if compact:
        return (shape[1], 1), _ceil(math.prod(shape), width) + width
    if transpose:
        pitch = _ceil(shape[0], width)
        return (1, pitch), shape[1] * pitch + width
    pitch = _ceil(shape[1], width)
    return (pitch, 1), shape[0] * pitch + width


def _output_shape(shape, dim, keepdims):
    if len(shape) == 1:
        return (1,)
    return tuple(1 if i == dim else n for i, n in enumerate(shape)) if keepdims else (shape[1 - dim],)


def _kernel(shape, dim, op, lanes, dtype, out_dtype, keepdims=False, transpose=False, compact=False, clear=True):
    strides, capacity = _storage(shape, lanes, dtype, transpose, compact)
    out_shape = _output_shape(shape, dim, keepdims)
    out_count = math.prod(out_shape)
    out_capacity = _ceil(out_count, lanes) + lanes
    reduce = getattr(T, "reduce_" + op)

    @T.prim_func
    def main(
        A: T.Tensor((capacity,), dtype),
        Initial: T.Tensor((out_capacity,), out_dtype),
        Output: T.Tensor((out_capacity,), out_dtype),
    ):
        with T.Kernel(1):
            source_owner = T.alloc_shared((capacity,), dtype)
            output_owner = T.alloc_shared((out_capacity,), out_dtype)
            source = T.StridedTensor(shape, strides, dtype, data=source_owner.data, scope="shared.dyn")
            destination = T.Tensor(out_shape, out_dtype, data=output_owner.data, scope="shared.dyn")
            T.copy(A, source_owner)
            T.copy(Initial, output_owner)
            with T.SimdVF(lanes=lanes):
                if op == "abssum":
                    reduce(source, destination, dim=dim)
                else:
                    reduce(source, destination, dim=dim, clear=clear)
            T.copy(output_owner, Output)

    return main


def _logical_values(shape, dtype, trial):
    size = math.prod(shape)
    if dtype == "uint32":
        values = np.resize(np.array([0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF], dtype=np.uint32), size)
        values = np.roll(values, trial)
        return torch.from_numpy(values.copy()).reshape(shape)
    if dtype == "int32":
        values = np.resize(np.array([-2147483648, 2147483647, -17, 0, 1, 37], dtype=np.int32), size)
        return torch.from_numpy(np.roll(values, trial).copy()).reshape(shape)
    # Dyadic values make different reduction orders well conditioned; changing
    # every trial detects stale carry/state while keeping f16 sums bounded.
    values = ((torch.arange(size, dtype=torch.int64) * (trial * 2 + 1) + trial * 3) % 31 - 15).float() / 8
    return values.to(TORCH_DTYPES[dtype]).reshape(shape)


def _reference(logical, dim, op, out_dtype):
    array = logical.to(torch.float64).numpy() if out_dtype.startswith("float") else logical.numpy()
    if op in ABS_OPS:
        # NumPy integer abs preserves the input width, including INT_MIN.
        array = np.abs(array)
    moved = np.moveaxis(array, dim, -1)
    result = []
    for vector in moved.reshape(-1, array.shape[dim]):
        if out_dtype in ("int32", "uint32"):
            values = [int(value) for value in vector]
            if op in ("sum", "abssum"):
                value = sum(values)
            elif op == "absmax":
                value = max(0, max(values))
            else:
                value = max(values) if op == "max" else min(values)
        else:
            value = (
                math.fsum(float(v) for v in vector)
                if op in ("sum", "abssum")
                else (max(vector) if op in ("max", "absmax") else min(vector))
            )
        if out_dtype in ("int32", "uint32"):
            value %= 1 << 32
            if out_dtype == "int32" and value >= 1 << 31:
                value -= 1 << 32
        result.append(value)
    return torch.tensor(result, dtype=TORCH_DTYPES[out_dtype])


def _case_inputs(shape, dim, op, lanes, dtype, out_dtype, transpose, trial, compact=False):
    logical = _logical_values(shape, dtype, trial)
    strides, capacity = _storage(shape, lanes, dtype, transpose, compact)
    padding = 9999 if dtype != "uint32" else 0xFFFFFFF0
    storage = torch.full((capacity,), padding, dtype=TORCH_DTYPES[dtype])
    # Padding values dwarf the valid values, exposing an incorrect final mask.
    if len(shape) == 1:
        storage[: shape[0]] = logical
    else:
        for i in range(shape[0]):
            for j in range(shape[1]):
                storage[i * strides[0] + j * strides[1]] = logical[i, j]
    out_count = math.prod(_output_shape(shape, dim, False))
    out_capacity = _ceil(out_count, lanes) + lanes
    sentinel = -777 if out_dtype != "uint32" else 0xBADF00D
    initial = torch.full((out_capacity,), sentinel, dtype=TORCH_DTYPES[out_dtype])
    expected = initial.clone()
    expected[:out_count] = _reference(logical, dim, op, out_dtype)
    return storage, initial, expected, out_count


def _run_case(npu, shape, dim, op, lanes=128, dtype="float32", out_dtype="float32", keepdims=False, transpose=False, compact=False):
    kernel = tilelang.compile(_kernel(shape, dim, op, lanes, dtype, out_dtype, keepdims, transpose, compact), target="pto", out_idx=-1)
    source = kernel.get_kernel_source()
    assert "pto.set_flag(" in source and "pto.wait_flag(" in source, "automatic MTE synchronization must stay enabled"
    assert "pto.vmi." in source
    for trial in range(2):
        a, initial, expected, count = _case_inputs(shape, dim, op, lanes, dtype, out_dtype, transpose, trial, compact)
        actual = kernel(a.to(npu), initial.to(npu))
        torch.npu.synchronize()
        got = actual.cpu()
        assert got.dtype == expected.dtype
        if out_dtype in ("int32", "uint32") or op not in ("sum", "abssum"):
            assert torch.equal(got[:count], expected[:count]), (shape, dim, op, dtype, out_dtype, got[:count], expected[:count])
        else:
            torch.testing.assert_close(
                got[:count], expected[:count], rtol=2e-3 if out_dtype == "float16" else 2e-5, atol=2e-2 if out_dtype == "float16" else 2e-5
            )
        assert torch.equal(got[count:], expected[count:]), "reduce modified destination canaries"


# Boundary cases are intentionally a table rather than an unrelated full
# product: every op covers all important E/VL relations at least once.
BOUNDARIES = [
    (1, 64),
    (63, 64),
    (64, 64),
    (65, 64),
    (127, 128),
    (128, 128),
    (129, 128),
    (259, 256),
    (4099, 128),
]


@pytest.mark.pto
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize(("extent", "lanes"), BOUNDARIES)
def test_direct_reduce_1d(npu, extent, lanes, op):
    _run_case(npu, (extent,), 0, op, lanes)


MATRIX_CASES = [
    ((3, 130), 1, 128, False, False),  # A: reduction tail
    ((3, 130), 0, 128, False, False),  # B: output tail
    ((5, 257), 1, 256, True, False),
    ((5, 257), 0, 256, True, False),
    ((1, 129), 1, 64, False, False),
    ((3, 1), 1, 128, False, False),
    ((129, 1), 0, 128, False, False),
    ((130, 3), 0, 128, False, True),  # A on physical outer logical axis
    ((130, 3), 1, 128, False, True),  # B on physical outer output axis
]


@pytest.mark.pto
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize(("shape", "dim", "lanes", "keepdims", "transpose"), MATRIX_CASES)
def test_direct_reduce_matrix(npu, shape, dim, lanes, keepdims, transpose, op):
    _run_case(npu, shape, dim, op, lanes, keepdims=keepdims, transpose=transpose)


@pytest.mark.pto
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("dim", [0, 1])
def test_direct_reduce_compact_row_pitch(npu, dim, op):
    # Row 1 begins at byte 520: natural alignment, not a 32B vector base.
    # The actual allocated owner still covers and initializes the full read.
    _run_case(npu, (3, 130), dim, op, compact=True)


DTYPE_CASES = [
    ((259,), 0, "float16", "float16"),
    ((3, 130), 1, "float16", "float32"),
    ((3, 130), 0, "float16", "float32"),
    ((3, 130), 1, "bfloat16", "float32"),
    ((3, 130), 0, "bfloat16", "float32"),
    ((3, 130), 1, "int32", "int32"),
    ((3, 130), 0, "int32", "int32"),
    ((3, 130), 1, "uint32", "uint32"),
    ((3, 130), 0, "uint32", "uint32"),
]


@pytest.mark.pto
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize(("shape", "dim", "dtype", "out_dtype"), DTYPE_CASES)
def test_direct_reduce_dtype(npu, shape, dim, dtype, out_dtype, op):
    _run_case(npu, shape, dim, op, dtype=dtype, out_dtype=out_dtype)


def _flow_kernel():
    @T.prim_func
    def main(A: T.Tensor((256,), "float32"), Initial: T.Tensor((256,), "float32"), Output: T.Tensor((256,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((256,), "float32")
            result = T.alloc_shared((256,), "float32")
            first = T.Tensor((1,), "float32", data=result.data, scope="shared.dyn")
            middle_owner = T.alloc_shared((256,), "float32")
            middle = T.Tensor((1,), "float32", data=middle_owner.data, scope="shared.dyn")
            last_owner = T.alloc_shared((256,), "float32")
            last = T.Tensor((1,), "float32", data=last_owner.data, scope="shared.dyn")
            T.copy(A, a)
            T.copy(Initial, result)
            with T.SimdVF(lanes=128):
                T.reduce_sum(a, first, dim=0)
                for i in T.Parallel(256):
                    a[i] = a[i] * T.float32(2)
                T.reduce_max(a, middle, dim=0)
                for i in T.Parallel(256):
                    a[i] = a[i] + middle[0]
                T.reduce_min(a, last, dim=0)
                # Ordinary UB scalar reads after prior 1PT result stores.
                result[1] = middle[0]
                result[2] = last[0]
            T.copy(result, Output)

    return main


@pytest.mark.pto
def test_direct_reduce_source_order_and_consumers(npu):
    kernel = tilelang.compile(_flow_kernel(), target="pto", out_idx=-1)
    for trial in range(2):
        a = _logical_values((256,), "float32", trial)
        initial = torch.full((256,), -777.0)
        expected = initial.clone()
        expected[0] = a.double().sum().float()
        expected[1] = (a * 2).max()
        expected[2] = (a * 2 + expected[1]).min()
        got = kernel(a.to(npu), initial.to(npu))
        torch.npu.synchronize()
        torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=2e-5)


def _serial_kernel(unrolled):
    @T.prim_func
    def main(A: T.Tensor((3, 256), "float32"), Output: T.Tensor((3,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((256,), "float32")
            b = T.alloc_shared((1,), "float32")
            for t in T.serial(3):
                T.copy(A[t, :], a)
                with T.SimdVF(lanes=128):
                    if unrolled:
                        for _repeat in T.Unroll(2):
                            T.reduce_sum(a, b, dim=0)
                    else:
                        for _repeat in T.serial(2):
                            T.reduce_sum(a, b, dim=0)
                T.copy(b, Output[t : t + 1])

    return main


@pytest.mark.pto
@pytest.mark.parametrize("unrolled", [False, True], ids=["serial", "unrolled"])
def test_direct_reduce_serial_wrapper(npu, unrolled):
    kernel = tilelang.compile(_serial_kernel(unrolled), target="pto", out_idx=-1)
    a = _logical_values((3, 256), "float32", 1)
    got = kernel(a.to(npu))
    torch.npu.synchronize()
    torch.testing.assert_close(got.cpu(), a.double().sum(dim=1).float(), rtol=0, atol=2e-5)


def _versioned_kernel(shape, dim, keepdims, source_versions, destination_versions):
    out_shape = _output_shape(shape, dim, keepdims)

    @T.prim_func
    def main(A: T.Tensor((4, *shape), "float32"), Output: T.Tensor((4, *out_shape), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared(shape, "float32")
            b = T.alloc_shared(out_shape, "float32")
            T.annotate_buffer_versions({a: source_versions, b: destination_versions})
            for t in T.Pipelined(4, num_stages=2):
                if len(shape) == 1:
                    T.copy(A[t, :], a)
                else:
                    T.copy(A[t, :, :], a)
                with T.SimdVF(lanes=128):
                    T.reduce_sum(a, b, dim=dim)
                if len(out_shape) == 1:
                    T.copy(b, Output[t, :])
                else:
                    T.copy(b, Output[t, :, :])

    return main


@pytest.mark.pto
@pytest.mark.parametrize(
    "shape, dim, keepdims, source_versions, destination_versions",
    [
        ((3, 128), 0, False, 2, 2),
        ((3, 128), 1, False, 2, 2),
        ((256,), 0, False, 2, 2),
        ((3, 128), 1, True, 2, 2),
        ((1, 128), 1, True, 1, 1),
        ((3, 128), 1, False, 2, 1),
        ((3, 128), 0, True, 1, 2),
        ((3, 128), 0, False, 3, 3),
        ((3, 128), 1, False, 3, 2),
    ],
)
def test_direct_reduce_multibuffer_logical_axis(npu, shape, dim, keepdims, source_versions, destination_versions):
    kernel = tilelang.compile(_versioned_kernel(shape, dim, keepdims, source_versions, destination_versions), target="pto", out_idx=-1)
    a = _logical_values((4, *shape), "float32", 1)
    got = kernel(a.to(npu))
    torch.npu.synchronize()
    expected = a.double().sum(dim=dim + 1, keepdim=keepdims).float()
    if len(shape) == 1:
        expected = expected.reshape(4, 1)
    torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=2e-5)


@pytest.mark.pto
@pytest.mark.parametrize("op", OPS + ABS_OPS)
@pytest.mark.parametrize("extent", [63, 259])
def test_direct_reduce_nan_padding(npu, extent, op):
    # NaNs occur only in initialized padding, never in the logical input. Both
    # the first-domain mask (E<VL) and keep-prior tail fold (E>VL) are visible.
    lanes = 128
    kernel = tilelang.compile(_kernel((extent,), 0, op, lanes, "float32", "float32"), target="pto", out_idx=-1)
    a, initial, expected, count = _case_inputs((extent,), 0, op, lanes, "float32", "float32", False, 1)
    a[extent:] = float("nan")
    got = kernel(a.to(npu), initial.to(npu))
    torch.npu.synchronize()
    torch.testing.assert_close(got.cpu()[:count], expected[:count], rtol=0, atol=2e-5)
    assert torch.equal(got.cpu()[count:], expected[count:])


@pytest.mark.pto
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("op", ["sum", "abssum"])
def test_direct_reduce_widen_before_accumulation(npu, dtype, dim, op):
    shape = (129, 130)
    lanes = 128
    strides, capacity = _storage(shape, lanes, dtype)
    logical = (torch.arange(math.prod(shape)).reshape(shape).float() % 43 * 0.00217 + 0.031).to(TORCH_DTYPES[dtype])
    expected_values = _reference(logical, dim, op, "float32")
    # A narrow reduction followed by a final cast must be observably different.
    assert (logical.sum(dim=dim).float().reshape(-1) - expected_values).abs().max() > 1e-3
    source = torch.full((capacity,), float("nan"), dtype=TORCH_DTYPES[dtype])
    source.as_strided(shape, strides).copy_(logical)
    out_count = expected_values.numel()
    initial = torch.full((_ceil(out_count, lanes) + lanes,), -777.0)
    kernel = tilelang.compile(_kernel(shape, dim, op, lanes, dtype, "float32"), target="pto", out_idx=-1)
    got = kernel(source.to(npu), initial.to(npu))
    torch.npu.synchronize()
    torch.testing.assert_close(got.cpu()[:out_count], expected_values, rtol=0, atol=2e-5)
    assert torch.equal(got.cpu()[out_count:], initial[out_count:])


def _versioned_tail_kernel(dim):
    shape = (3, 130)
    lanes = 128
    strides, capacity = _storage(shape, lanes, "float32")
    out_count = shape[1 - dim]
    out_capacity = _ceil(out_count, lanes) + lanes

    @T.prim_func
    def main(
        A: T.Tensor((4, capacity), "float32"),
        Initial: T.Tensor((4, out_capacity), "float32"),
        Output: T.Tensor((4, out_capacity), "float32"),
    ):
        with T.Kernel(1):
            source_owner = T.alloc_shared((capacity,), "float32")
            output_owner = T.alloc_shared((out_capacity,), "float32")
            source = T.StridedTensor(shape, strides, "float32", data=source_owner.data, scope="shared.dyn")
            destination = T.Tensor((out_count,), "float32", data=output_owner.data, scope="shared.dyn")
            T.annotate_buffer_versions({source_owner: 2, output_owner: 2})
            for t in T.Pipelined(4, num_stages=2):
                T.copy(A[t, :], source_owner)
                T.copy(Initial[t, :], output_owner)
                with T.SimdVF(lanes=lanes):
                    T.reduce_sum(source, destination, dim=dim)
                T.copy(output_owner, Output[t, :])

    return main


@pytest.mark.pto
@pytest.mark.parametrize("dim", [0, 1])
def test_direct_reduce_multibuffer_tail(npu, dim):
    cases = [_case_inputs((3, 130), dim, "sum", 128, "float32", "float32", False, trial) for trial in range(4)]
    source = torch.stack([case[0] for case in cases])
    initial = torch.stack([case[1] for case in cases])
    expected = torch.stack([case[2] for case in cases])
    kernel = tilelang.compile(_versioned_tail_kernel(dim), target="pto", out_idx=-1)
    got = kernel(source.to(npu), initial.to(npu))
    torch.npu.synchronize()
    torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=2e-5)


def _accumulate_kernel(shape, dim, op, lanes, dtype, out_dtype, keepdims, repeats):
    strides, capacity = _storage(shape, lanes, dtype)
    out_shape = _output_shape(shape, dim, keepdims)
    reduce = getattr(T, "reduce_" + op)

    @T.prim_func
    def main(
        A: T.Tensor((capacity,), dtype),
        Initial: T.Tensor(out_shape, out_dtype),
        Output: T.Tensor(out_shape, out_dtype),
    ):
        with T.Kernel(1):
            owner = T.alloc_shared((capacity,), dtype)
            source = T.StridedTensor(shape, strides, dtype, data=owner.data, scope="shared.dyn")
            destination = T.alloc_shared(out_shape, out_dtype)
            T.copy(A, owner)
            T.copy(Initial, destination)
            with T.SimdVF(lanes=lanes):
                for _repeat in T.serial(repeats):
                    reduce(source, destination, dim=dim, clear=False)
            T.copy(destination, Output)

    return main


def _combine_previous(previous, reduced, op, dtype):
    if dtype in ("int32", "uint32"):
        combined = []
        for a, b in zip(previous.reshape(-1).tolist(), reduced.reshape(-1).tolist()):
            value = int(a) + int(b) if op == "sum" else max(int(a), int(b)) if op in ("max", "absmax") else min(int(a), int(b))
            value %= 1 << 32
            if dtype == "int32" and value >= 1 << 31:
                value -= 1 << 32
            combined.append(value)
        return torch.tensor(combined, dtype=TORCH_DTYPES[dtype]).reshape(previous.shape)
    if op == "sum":
        return (previous.double() + reduced.double()).to(TORCH_DTYPES[dtype])
    return torch.maximum(previous, reduced) if op in ("max", "absmax") else torch.minimum(previous, reduced)


ACCUMULATE_CASES = [
    ((1,), 0, 64, False, 1),
    ((63,), 0, 64, False, 2),
    ((259,), 0, 128, False, 2),
    ((3, 256), 1, 64, False, 1),
    ((3, 130), 1, 128, True, 2),
    ((3, 130), 0, 128, False, 2),
    ((3, 3), 0, 128, True, 1),
    ((1, 257), 0, 256, False, 1),
]


@pytest.mark.pto
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("shape, dim, lanes, keepdims, repeats", ACCUMULATE_CASES)
def test_direct_reduce_accumulates_previous_output(npu, shape, dim, lanes, keepdims, repeats, op):
    _run_accumulate(npu, shape, dim, op, lanes, "float32", "float32", keepdims, repeats)


def _run_accumulate(npu, shape, dim, op, lanes, dtype, out_dtype, keepdims=False, repeats=2):
    kernel = tilelang.compile(_accumulate_kernel(shape, dim, op, lanes, dtype, out_dtype, keepdims, repeats), target="pto", out_idx=-1)
    strides, capacity = _storage(shape, lanes, dtype)
    out_shape = _output_shape(shape, dim, keepdims)
    count = math.prod(out_shape)
    for trial in range(2):
        logical = _logical_values(shape, dtype, trial)
        source = torch.zeros((capacity,), dtype=TORCH_DTYPES[dtype])
        source.as_strided(shape, strides).copy_(logical)
        if out_dtype == "uint32":
            initial = torch.tensor([0xFFFFFFF0 if i % 2 else 1 for i in range(count)], dtype=torch.uint32).reshape(out_shape)
        else:
            initial = torch.tensor([100 if (i + trial) % 2 else -100 for i in range(count)], dtype=TORCH_DTYPES[out_dtype]).reshape(
                out_shape
            )
        reduced = _reference(logical, dim, op, out_dtype).reshape(out_shape)
        expected = initial.clone()
        for _ in range(repeats):
            expected = _combine_previous(expected, reduced, op, out_dtype)
        got = kernel(source.to(npu), initial.to(npu))
        torch.npu.synchronize()
        if out_dtype in ("int32", "uint32") or op not in ("sum", "abssum"):
            assert torch.equal(got.cpu(), expected)
        else:
            torch.testing.assert_close(
                got.cpu(), expected, rtol=2e-3 if out_dtype == "float16" else 2e-5, atol=2e-2 if out_dtype == "float16" else 2e-5
            )


@pytest.mark.pto
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize(
    "dtype, out_dtype",
    [("float16", "float16"), ("float16", "float32"), ("bfloat16", "float32"), ("int32", "int32"), ("uint32", "uint32")],
)
def test_direct_reduce_accumulate_dtype(npu, dim, dtype, out_dtype, op):
    _run_accumulate(npu, (3, 130), dim, op, 128, dtype, out_dtype)


@pytest.mark.pto
@pytest.mark.parametrize("columns", [64, 128])
@pytest.mark.parametrize("op", OPS)
def test_direct_reduce_accumulate_short_fp16_read_width(npu, columns, op):
    # VL64 is narrower than one physical FP16 read. Both chunks still use
    # vector loads; only the active logical lanes participate in the fold
    # and store, even when the physical load extends past the logical shape.
    _run_accumulate(npu, (3, columns), 0, op, 64, "float16", "float16")


@pytest.mark.pto
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("op", OPS)
def test_direct_reduce_accumulate_versioned_tail(npu, dim, op):
    count = 3 if dim == 1 else 130
    reduce = getattr(T, "reduce_" + op)

    @T.prim_func
    def main(A: T.Tensor((4, 3, 256), "float32"), Initial: T.Tensor((4, count), "float32"), B: T.Tensor((4, count), "float32")):
        with T.Kernel(1):
            owner = T.alloc_shared((3, 256), "float32")
            source = T.StridedTensor((3, 130), (256, 1), "float32", data=owner.data, scope="shared.dyn")
            destination = T.alloc_shared((count,), "float32")
            T.annotate_buffer_versions({owner: 2, destination: 2})
            for t in T.Pipelined(4, num_stages=2):
                T.copy(A[t, :, :], owner)
                T.copy(Initial[t, :], destination)
                with T.SimdVF(lanes=128):
                    reduce(source, destination, dim=dim, clear=False)
                T.copy(destination, B[t, :])

    logical = _logical_values((4, 3, 130), "float32", 1)
    source = torch.zeros((4, 3, 256))
    source[:, :, :130] = logical
    initial = (torch.arange(4 * count).reshape(4, count) % 11 - 5).float()
    if op == "sum":
        expected = initial + logical.double().sum(dim=dim + 1).float()
    else:
        reduced = logical.max(dim=dim + 1).values if op == "max" else logical.min(dim=dim + 1).values
        expected = torch.maximum(initial, reduced) if op == "max" else torch.minimum(initial, reduced)
    kernel = tilelang.compile(main, target="pto", out_idx=-1)
    got = kernel(source.to(npu), initial.to(npu))
    torch.npu.synchronize()
    torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=2e-5)


@pytest.mark.pto
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize(
    "dim, lanes, clear",
    [(1, 64, True), (1, 128, False), (1, 256, True), (0, 64, False), (0, 128, True), (0, 256, False)],
    ids=["a-vl64-clear", "a-vl128-accumulate", "a-vl256-clear", "b-vl64-accumulate", "b-vl128-clear", "b-vl256-accumulate"],
)
def test_direct_reduce_compact_full_matrix(npu, dim, lanes, clear, op):
    """Logical and allocated shapes are exactly (3, 256), with no padding."""
    count = 3 if dim == 1 else 256
    reduce = getattr(T, "reduce_" + op)

    @T.prim_func
    def main(A: T.Tensor((3, 256), "float32"), Initial: T.Tensor((count,), "float32"), B: T.Tensor((count,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((3, 256), "float32")
            b = T.alloc_shared((count,), "float32")
            T.copy(A, a)
            T.copy(Initial, b)
            with T.SimdVF(lanes=lanes):
                reduce(a, b, dim=dim, clear=clear)
            T.copy(b, B)

    kernel = tilelang.compile(main, target="pto", out_idx=-1)
    generated = kernel.get_kernel_source()
    assert ("pto.vmi.vc" in generated) == (dim == 1)
    for trial in range(2):
        logical = _logical_values((3, 256), "float32", trial)
        initial = ((torch.arange(count) * 7 + trial) % 19 - 9).float()
        reduced = _reference(logical, dim, op, "float32")
        expected = reduced if clear else _combine_previous(initial, reduced, op, "float32")
        got = kernel(logical.to(npu), initial.to(npu))
        torch.npu.synchronize()
        torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=2e-5)


@pytest.mark.pto
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("clear", [False, True], ids=["accumulate", "clear"])
def test_direct_reduce_preserves_view_offsets(npu, dim, clear):
    """A natural-aligned source offset and aligned output offset count once."""
    count = 3 if dim == 1 else 130
    source_offset, destination_offset = 3, 8
    source_capacity, destination_capacity = 1024, 264

    @T.prim_func
    def main(
        A: T.Tensor((source_capacity,), "float32"),
        Initial: T.Tensor((destination_capacity,), "float32"),
        B: T.Tensor((destination_capacity,), "float32"),
    ):
        with T.Kernel(1):
            source_owner = T.alloc_shared((source_capacity,), "float32")
            destination_owner = T.alloc_shared((destination_capacity,), "float32")
            source = tilelang.tvm.tirx.decl_buffer(
                (3, 130), "float32", data=source_owner.data, strides=(256, 1), elem_offset=source_offset, scope="shared.dyn"
            )
            destination = tilelang.tvm.tirx.decl_buffer(
                (count,), "float32", data=destination_owner.data, elem_offset=destination_offset, scope="shared.dyn"
            )
            T.copy(A, source_owner)
            T.copy(Initial, destination_owner)
            with T.SimdVF(lanes=128):
                T.reduce_sum(source, destination, dim=dim, clear=clear)
            T.copy(destination_owner, B)

    kernel = tilelang.compile(main, target="pto", out_idx=-1)
    for trial in range(2):
        logical = _logical_values((3, 130), "float32", trial)
        source = torch.full((source_capacity,), float("nan"))
        source.as_strided((3, 130), (256, 1), source_offset).copy_(logical)
        initial = ((torch.arange(destination_capacity) * 7 + trial) % 19 - 9).float()
        expected = initial.clone()
        reduced = _reference(logical, dim, "sum", "float32")
        selection = slice(destination_offset, destination_offset + count)
        expected[selection] = reduced if clear else initial[selection] + reduced
        got = kernel(source.to(npu), initial.to(npu))
        torch.npu.synchronize()
        torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=2e-5)


def _make_pto_reduce_reuse_program(
    extent=130,
    lanes=128,
    dim=1,
    dtype="float32",
    out_dtype="float32",
    clear=True,
    versions=1,
    interleave=False,
    keep_padding=False,
    source_offset=0,
    complete_read=True,
):
    """Allocate the complete physical read, then test storage lifetime reuse.

    The strided allocation itself owns its row padding. The owner alias is used
    only to initialize that capacity; reduce still receives a complete logical
    Buffer. An ordinary scalar result read completes the VF before later is
    overwritten, while its DMA consumer independently exercises ring flags.
    """
    from tilelang import tvm

    rows = 1 if dim == 1 else 3
    width = max(lanes, 2048 // tvm.DataType(dtype).bits)
    pitch = (extent + source_offset + width - 1) // width * width
    if source_offset:
        # Leave a complete extra physical vector for the last row's offset.
        pitch += width
    output_count = rows if dim == 1 else extent

    @T.prim_func
    def main(
        A: T.Tensor((4,), dtype),
        B: T.Tensor((4,), dtype),
        Initial: T.Tensor((4,), out_dtype),
        C: T.Tensor((4, output_count + 2), out_dtype),
    ):
        with T.Kernel(1):
            a = T.sblock_alloc_buffer((rows, extent), dtype, strides=(pitch, 1), elem_offset=source_offset, scope="shared.dyn")
            owner = T.StridedTensor((rows, pitch), (pitch, 1), dtype, data=a.data, scope="shared.dyn")
            result = T.alloc_shared((output_count,), out_dtype)
            later = T.alloc_shared((rows, pitch), dtype)
            if versions > 1:
                T.annotate_buffer_versions({a: versions, later: versions})
            if keep_padding:
                T.fill(owner, T.cast(-999, dtype))
            for t in T.Pipelined(4, num_stages=2):
                with T.Stage(0):
                    if not keep_padding:
                        T.fill(owner, T.cast(-999, dtype))
                with T.Stage(0):
                    T.fill(a, A[t])
                with T.Stage(0):
                    T.fill(result, Initial[t])
                with T.Stage(0):
                    if interleave:
                        T.fill(later, B[t])
                with T.Stage(0), T.SimdVF(lanes=lanes):
                    T.reduce_sum(a, result, dim=dim, clear=clear)
                with T.Stage(0):
                    T.copy(result, C[t, 0:output_count])
                with T.Stage(0):
                    if complete_read:
                        C[t, output_count] = result[0]
                with T.Stage(0):
                    if not interleave:
                        T.fill(later, B[t])
                with T.Stage(0):
                    C[t, output_count + 1] = later[0, 0]
                with T.Stage(0):
                    if not complete_read:
                        C[t, output_count] = result[0]

    return main


@pytest.mark.pto
@pytest.mark.parametrize(
    "options",
    [
        {"extent": 128},
        {},
        {"extent": 3},
        {"extent": 63, "lanes": 64, "dtype": "float16", "out_dtype": "float16"},
        {"dim": 0},
        {"clear": False},
        {"versions": 2},
        {"versions": 3},
        {"dim": 0, "versions": 2},
        {"source_offset": 3},
        {"interleave": True},
        {"keep_padding": True},
        {"complete_read": False},
        {"complete_read": False, "versions": 2},
    ],
    ids=[
        "full",
        "tail",
        "short",
        "physical-read-width",
        "independent-outputs",
        "old-output",
        "two-versions",
        "three-versions",
        "versioned-outputs",
        "offset",
        "interleaved-source",
        "live-in-padding",
        "uncompleted-vf",
        "uncompleted-versioned-vf",
    ],
)
def test_direct_reduce_storage_reuse_preserves_values(npu, options):
    dtype = options.get("dtype", "float32")
    out_dtype = options.get("out_dtype", "float32")
    dim = options.get("dim", 1)
    extent = options.get("extent", 130)
    count = 1 if dim == 1 else extent
    kernel = tilelang.compile(_make_pto_reduce_reuse_program(**options), target="pto", out_idx=-1)
    for trial in range(2):
        a = (torch.tensor([1, -2, 3, 4]) + trial).to(TORCH_DTYPES[dtype])
        b = (torch.tensor([-11, 12, -13, 14]) - trial).to(TORCH_DTYPES[dtype])
        initial = (torch.tensor([19, -7, 23, -9]) + trial).to(TORCH_DTYPES[out_dtype])
        reduced = a.to(TORCH_DTYPES[out_dtype]) * (extent if dim == 1 else 3)
        if not options.get("clear", True):
            reduced = reduced + initial
        expected = torch.empty((4, count + 2), dtype=TORCH_DTYPES[out_dtype])
        expected[:, :count] = reduced[:, None]
        expected[:, count] = reduced
        expected[:, count + 1] = b.to(TORCH_DTYPES[out_dtype])
        actual = kernel(a.to(npu), b.to(npu), initial.to(npu))
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.pto
@pytest.mark.parametrize(
    "group, outputs, dim, lanes, dtype, clear",
    [
        (16, 3, 1, 64, "float32", True),
        (32, 8, 1, 128, "float16", True),
        (64, 3, 1, 64, "float32", False),
        (128, 3, 1, 256, "float16", False),
        (16, 128, 0, 64, "float32", True),
        (32, 256, 0, 128, "float16", False),
        (64, 130, 0, 128, "float16", True),
        (64, 64, 0, 256, "float32", False),
    ],
)
def test_direct_reduce_group_absmax_and_immediate_consumer(npu, group, outputs, dim, lanes, dtype, clear):
    """Group absmax with input preparation and an immediate elementwise epilogue."""
    shape = (outputs, group) if dim == 1 else (group, outputs)
    pitch = _ceil(shape[1], max(lanes, 128))

    @T.prim_func
    def main(A: T.Tensor((shape[0], pitch), dtype), Initial: T.Tensor((outputs,), "float32"), B: T.Tensor((outputs,), "float32")):
        with T.Kernel(1):
            owner = T.alloc_shared((shape[0], pitch), dtype)
            source = T.StridedTensor(shape, (pitch, 1), dtype, data=owner.data, scope="shared.dyn")
            prepared = T.alloc_shared((shape[0], pitch), "float32")
            absolute = T.StridedTensor(shape, (pitch, 1), "float32", data=prepared.data, scope="shared.dyn")
            result = T.alloc_shared((outputs,), "float32")
            T.copy(A, owner)
            T.fill(prepared, T.float32(0))
            T.copy(Initial, result)
            with T.SimdVF(lanes=lanes):
                for i, j in T.Parallel(shape[0], shape[1]):
                    absolute[i, j] = T.abs(T.Cast("float32", source[i, j]))
                T.reduce_max(absolute, result, dim=dim, clear=clear)
                for i in T.Parallel(outputs):
                    result[i] = result[i] + T.float32(1)
            T.copy(result, B)

    kernel = tilelang.compile(main, target="pto", out_idx=-1)
    for trial in range(2):
        logical = _logical_values(shape, dtype, trial)
        initial = torch.full((shape[0], pitch), float("nan"), dtype=TORCH_DTYPES[dtype])
        initial[:, : shape[1]] = logical
        previous = ((torch.arange(outputs) + trial) % 3).float() * 2
        reduced = logical.float().abs().amax(dim=dim)
        expected = (reduced if clear else torch.maximum(reduced, previous)) + 1
        got = kernel(initial.to(npu), previous.to(npu))
        torch.npu.synchronize()
        torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)


@pytest.mark.pto
@pytest.mark.parametrize("lanes, dtype", [(64, "float32"), (128, "float16"), (128, "bfloat16")])
def test_direct_reduce_hierarchy_and_fill_consumer(npu, lanes, dtype):
    """Reduce a published partial result, then broadcast it to a fill consumer."""

    @T.prim_func
    def main(A: T.Tensor((128, 128), dtype), B: T.Tensor((2, 128), "float32")):
        with T.Kernel(1):
            source = T.alloc_shared((128, 128), dtype)
            absolute = T.alloc_shared((128, 128), "float32")
            partial = T.alloc_shared((128,), "float32")
            minimum = T.alloc_shared((1,), "float32")
            result = T.alloc_shared((128,), "float32")
            T.copy(A, source)
            for _repeat in T.serial(2):
                with T.SimdVF(lanes=lanes):
                    T.clear(result)
                    for i, j in T.Parallel(128, 128):
                        absolute[i, j] = T.abs(T.Cast("float32", source[i, j]))
                    T.reduce_max(absolute, partial, dim=1)
                    T.reduce_min(partial, minimum, dim=0)
                    T.fill(result, minimum[0])
                    for i in T.Parallel(128):
                        result[i] = result[i] + partial[i]
                T.copy(result, B[_repeat, :])

    kernel = tilelang.compile(main, target="pto", out_idx=-1)
    for trial in range(2):
        row_scale = ((torch.arange(128) + 3 * trial) % 7 + 1 + trial).float() / 8
        initial = (_logical_values((128, 128), dtype, trial).float() * row_scale[:, None]).to(TORCH_DTYPES[dtype])
        partial = initial.float().abs().amax(dim=1)
        expected = (partial + partial.min()).expand(2, 128)
        got = kernel(initial.to(npu))
        torch.npu.synchronize()
        torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)


# Each new operation exercises the two physical reduction directions, tails,
# and supported dtype routes without multiplying the complete legacy matrix.
@pytest.mark.pto
@pytest.mark.parametrize("op", ABS_OPS)
@pytest.mark.parametrize(
    "shape, dim, lanes, dtype, out_dtype, transpose",
    [
        ((63,), 0, 64, "float32", "float32", False),
        ((259,), 0, 256, "float16", "float16", False),
        ((3, 130), 1, 128, "float16", "float32", False),
        ((3, 130), 0, 128, "float16", "float32", False),
        ((3, 130), 1, 128, "bfloat16", "float32", False),
        ((3, 130), 0, 128, "bfloat16", "float32", False),
        ((130, 3), 0, 128, "float32", "float32", True),
        ((130, 3), 1, 128, "float32", "float32", True),
        ((3, 130), 1, 128, "int32", "int32", False),
        ((3, 130), 0, 128, "int32", "int32", False),
        ((3, 130), 1, 128, "uint32", "uint32", False),
        ((3, 130), 0, 128, "uint32", "uint32", False),
    ],
)
def test_direct_reduce_absolute(npu, op, shape, dim, lanes, dtype, out_dtype, transpose):
    _run_case(npu, shape, dim, op, lanes, dtype, out_dtype, keepdims=True, transpose=transpose)


@pytest.mark.pto
@pytest.mark.parametrize("dtype, out_dtype", [("float32", "float32"), ("float16", "float32"), ("int32", "int32"), ("uint32", "uint32")])
@pytest.mark.parametrize("dim", [0, 1])
def test_direct_reduce_absmax_accumulates_once(npu, dtype, out_dtype, dim):
    # Negative previous values must remain signed; taking abs(Initial) changes
    # the expected result for every negative float initial value in this case.
    _run_accumulate(npu, (3, 130), dim, "absmax", 128, dtype, out_dtype)


@pytest.mark.pto
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("shape", [(3, 65), (3, 259)], ids=["short", "chunks-tail"])
@pytest.mark.parametrize(
    "op, clear, initial_value",
    [("abssum", True, -777), ("absmax", True, -777), ("absmax", False, -777), ("absmax", False, 777)],
)
def test_direct_reduce_absolute_int_min(npu, dim, shape, op, clear, initial_value):
    lanes = 128
    strides, capacity = _storage(shape, lanes, "int32")
    logical = torch.full(shape, -(1 << 31), dtype=torch.int32)
    source = torch.full((capacity,), 9999, dtype=torch.int32)
    source.as_strided(shape, strides).copy_(logical)
    count = shape[1 - dim]
    initial = torch.full((_ceil(count, lanes) + lanes,), initial_value, dtype=torch.int32)
    expected = initial.clone()
    reduced = _reference(logical, dim, op, "int32")
    # Like Ascend/CUDA, absmax reduces from zero even with clear=False,
    # then merges the previous output without taking its absolute value.
    expected[:count] = reduced if clear else _combine_previous(initial[:count], reduced, op, "int32")
    kernel = tilelang.compile(_kernel(shape, dim, op, lanes, "int32", "int32", clear=clear), target="pto", out_idx=-1)
    got = kernel(source.to(npu), initial.to(npu))
    torch.npu.synchronize()
    assert torch.equal(got.cpu(), expected)
