"""Positive device coverage for high-level PTO UB fill/clear.

Every program uses the native Verify/Vectorize pipeline, with automatic
scheduling and no replacement hooks. CPU references include the complete
allocation so origin-zero subregions and stride gaps are checked together.
"""

from __future__ import annotations

import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.ascend.language as T


@pytest.fixture(scope="module")
def npu():
    return torch.device("npu")


def _kernel(shape, extents, dtype, lanes, clear, transpose=False):
    pitch = (shape[0] + 31) // 32 * 32 if transpose else (shape[-1] + 31) // 32 * 32
    strides = (1,) if len(shape) == 1 else (1, pitch) if transpose else (pitch, 1)
    capacity = shape[0] + 32 if len(shape) == 1 else pitch * shape[1 if transpose else 0] + 32
    value = 3.25 if dtype in ("float32", "float16", "bfloat16") else 37

    @T.prim_func
    def main(A: T.Tensor((capacity,), dtype), B: T.Tensor((capacity,), dtype)):
        with T.Kernel(1):
            owner = T.alloc_shared((capacity,), dtype)
            b = T.StridedTensor(shape, strides, dtype, data=owner.data, scope="shared.dyn")
            T.copy(A, owner)
            with T.SimdVF(lanes=lanes):
                if len(shape) == 1:
                    if clear:
                        T.clear(b[0 : extents[0]])
                    else:
                        T.fill(b[0 : extents[0]], value)
                else:
                    if clear:
                        T.clear(b[0 : extents[0], 0 : extents[1]])
                    else:
                        T.fill(b[0 : extents[0], 0 : extents[1]], value)
            T.copy(owner, B)

    return main, strides, capacity, value


REGIONS = [
    ((64,), (64,), 64, False),
    ((256,), (128,), 64, False),
    ((256,), (256,), 128, False),
    ((512,), (512,), 256, False),
    ((3, 256), (3, 256), 128, False),
    ((3, 384), (2, 256), 128, False),
    ((256, 3), (256, 3), 128, True),
    ((256, 3), (128, 2), 64, True),
]


@pytest.mark.pto
@pytest.mark.parametrize("clear", [False, True], ids=["fill", "clear"])
@pytest.mark.parametrize("shape, extents, lanes, transpose", REGIONS)
def test_fill_clear_regions(npu, shape, extents, lanes, transpose, clear):
    func, strides, capacity, value = _kernel(shape, extents, "float32", lanes, clear, transpose)
    kernel = tilelang.compile(func, target="pto", out_idx=-1)
    for trial in range(2):
        initial = torch.arange(capacity).float() / 8 + 11 + trial
        expected = initial.clone()
        view = expected.as_strided(shape, strides)
        slices = tuple(slice(0, extent) for extent in extents)
        view[slices] = 0 if clear else value
        got = kernel(initial.to(npu))
        torch.npu.synchronize()
        torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)


DTYPES = ["float16", "bfloat16", "int8", "uint8", "int16", "uint16", "int32", "uint32"]


@pytest.mark.pto
@pytest.mark.parametrize(
    "dtype, lanes",
    # Cover every dtype, then vary register packing for 8/16/32-bit types.
    [(dtype, 64) for dtype in DTYPES] + [(dtype, lanes) for dtype in ("int8", "float16", "int32") for lanes in (128, 256)],
)
@pytest.mark.parametrize("clear", [False, True], ids=["fill", "clear"])
def test_fill_clear_dtype(npu, dtype, lanes, clear):
    func, _, capacity, value = _kernel((256,), (256,), dtype, lanes, clear)
    initial = ((torch.arange(capacity) % 31) + 10).to(getattr(torch, dtype))
    expected = initial.clone()
    expected[:256] = 0 if clear else value
    kernel = tilelang.compile(func, target="pto", out_idx=-1)
    got = kernel(initial.to(npu))
    torch.npu.synchronize()
    assert torch.equal(got.cpu(), expected)


@pytest.mark.pto
@pytest.mark.parametrize("dtype", ["float32", *DTYPES])
def test_fill_captures_aliased_ub_value(npu, dtype):
    @T.prim_func
    def main(A: T.Tensor((256,), dtype), B: T.Tensor((2, 256), dtype)):
        with T.Kernel(1):
            b = T.alloc_shared((256,), dtype)
            T.copy(A, b)
            with T.SimdVF(lanes=64):
                T.fill(b, b[70])
            T.copy(b, B[0, :])
            with T.SimdVF(lanes=64):
                T.clear(b)
                T.fill(b, b[70])
            T.copy(b, B[1, :])

    initial = ((torch.arange(256) % 31) + 1).to(getattr(torch, dtype))
    expected = torch.empty((2, 256), dtype=initial.dtype)
    expected[0] = initial[70].item()
    expected[1] = 0
    kernel = tilelang.compile(main, target="pto", out_idx=-1)
    got = kernel(initial.to(npu))
    torch.npu.synchronize()
    assert torch.equal(got.cpu(), expected)


@pytest.mark.pto
@pytest.mark.parametrize("versions", [1, 2])
@pytest.mark.parametrize("columns", [128, 256])
@pytest.mark.parametrize("clear", [False, True], ids=["fill", "clear"])
def test_fill_clear_multibuffer(npu, versions, columns, clear):
    @T.prim_func
    def main(A: T.Tensor((4, 3, columns), "float32"), B: T.Tensor((4, 3, columns), "float32")):
        with T.Kernel(1):
            b = T.alloc_shared((3, columns), "float32")
            T.annotate_buffer_versions({b: versions})
            for t in T.Pipelined(4, num_stages=2):
                T.copy(A[t, :, :], b)
                with T.SimdVF(lanes=128):
                    if clear:
                        T.clear(b)
                    else:
                        T.fill(b, b[0, 0])
                T.copy(b, B[t, :, :])

    initial = torch.arange(4 * 3 * columns).reshape(4, 3, columns).float() / 8
    expected = torch.zeros_like(initial) if clear else initial[:, 0:1, 0:1].expand_as(initial)
    kernel = tilelang.compile(main, target="pto", out_idx=-1)
    got = kernel(initial.to(npu))
    torch.npu.synchronize()
    torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)


@pytest.mark.pto
def test_fill_clear_order_with_parallel_consumers(npu):
    @T.prim_func
    def main(B: T.Tensor((256,), "float32")):
        with T.Kernel(1):
            b = T.alloc_shared((256,), "float32")
            c = T.alloc_shared((256,), "float32")
            with T.SimdVF(lanes=128):
                T.fill(b, 3.0)
                for i in T.Parallel(256):
                    c[i] = b[i] * T.float32(2)
                T.clear(b)
                for i in T.Parallel(256):
                    b[i] = b[i] + c[i] + T.float32(1)
            T.copy(b, B)

    kernel = tilelang.compile(main, target="pto", out_idx=-1)
    got = kernel()
    torch.npu.synchronize()
    expected = torch.full((256,), 7.0)
    torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)


@pytest.mark.pto
@pytest.mark.parametrize(
    "dtype, bind_cast",
    [("float32", False), ("float16", False), ("bfloat16", False), ("float16", True)],
    ids=["direct-fp32", "direct-fp16", "direct-bf16", "bound-cast-fp16"],
)
def test_fill_runtime_scalar_parameter(npu, dtype, bind_cast):
    @T.prim_func
    def main(value: T.float32, B: T.Tensor((256,), dtype)):
        with T.Kernel(1):
            b = T.alloc_shared((256,), dtype)
            with T.SimdVF(lanes=64):
                if bind_cast:
                    snapshot = T.bind(T.Cast(dtype, value))
                    T.fill(b, snapshot)
                else:
                    T.fill(b, value)
            T.copy(b, B)

    kernel = tilelang.compile(main, target="pto", out_idx=-1)
    for value in [1.0003, -7.003]:
        got = kernel(value)
        torch.npu.synchronize()
        torch.testing.assert_close(got.cpu(), torch.full((256,), value, dtype=getattr(torch, dtype)), rtol=0, atol=0)


@pytest.mark.pto
@pytest.mark.parametrize("versions", [1, 2])
def test_fill_copy_nested_loops_runtime(npu, versions):
    @T.prim_func
    def main(outer_value: T.float32, inner_value: T.float32, B: T.Tensor((16, 17, 64), "float32")):
        with T.Kernel(1):
            b = T.alloc_shared((64,), "float32")
            T.annotate_buffer_versions({b: versions})
            for epoch in T.serial(16):
                with T.SimdVF(lanes=64):
                    T.fill(b, outer_value)
                T.copy(b, B[epoch, 0, :])
                for tile in T.serial(16):
                    with T.SimdVF(lanes=64):
                        T.fill(b, inner_value)
                    T.copy(b, B[epoch, tile + 1, :])

    kernel = tilelang.compile(main, target="pto", out_idx=-1)
    for outer_value, inner_value in [(3.25, -7.5), (-0.125, 1.0003)]:
        expected = torch.full((16, 17, 64), inner_value)
        expected[:, 0, :] = outer_value
        got = kernel(outer_value, inner_value)
        torch.npu.synchronize()
        torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)


@pytest.mark.pto
@pytest.mark.parametrize("reverse", [False, True], ids=["block-index", "reversed-block-index"])
def test_fill_uses_uniform_block_index(npu, reverse):
    @T.prim_func
    def main(A: T.Tensor((2,), "float32"), B: T.Tensor((2, 128), "float32")):
        with T.Kernel(2) as bx:
            values = T.alloc_shared((2,), "float32")
            b = T.alloc_shared((128,), "float32")
            T.copy(A, values)
            with T.SimdVF(lanes=128):
                if reverse:
                    T.fill(b, values[1 - bx])
                else:
                    T.fill(b, values[bx])
            T.copy(b, B[bx, :])

    kernel = tilelang.compile(main, target="pto", out_idx=-1)
    for values in [[3.25, -7.5], [-1.5, 11.25]]:
        initial = torch.tensor(values)
        expected = (initial.flip(0) if reverse else initial)[:, None].expand(2, 128)
        got = kernel(initial.to(npu))
        torch.npu.synchronize()
        torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)


@pytest.mark.pto
@pytest.mark.parametrize(
    "source_dtype, dtype, intermediate",
    [
        ("float16", "float32", None),
        ("bfloat16", "float32", None),
        ("float32", "float16", None),
        ("int32", "int8", None),
        ("float16", "bfloat16", "float32"),
    ],
)
def test_fill_converts_ub_snapshot_before_broadcast(npu, source_dtype, dtype, intermediate):
    @T.prim_func
    def main(A: T.Tensor((1,), source_dtype), B: T.Tensor((256,), dtype)):
        with T.Kernel(1):
            source = T.alloc_shared((1,), source_dtype)
            b = T.alloc_shared((256,), dtype)
            T.copy(A, source)
            with T.SimdVF(lanes=64):
                if intermediate is None:
                    T.fill(b, source[0])
                else:
                    T.fill(b, T.Cast(dtype, T.Cast(intermediate, source[0])))
            T.copy(b, B)

    kernel = tilelang.compile(main, target="pto", out_idx=-1)
    values = [1.0003, -7.003] if intermediate else [3.25 if "float" in source_dtype else 37]
    for value in values:
        initial = torch.tensor([value], dtype=getattr(torch, source_dtype))
        expected = initial.to(getattr(torch, intermediate)) if intermediate else initial
        expected = expected.to(getattr(torch, dtype)).expand(256)
        got = kernel(initial.to(npu))
        torch.npu.synchronize()
        assert torch.equal(got.cpu(), expected)


@pytest.mark.pto
@pytest.mark.parametrize(
    "intermediate, lanes, parameter",
    [
        pytest.param("float16", 64, False, id="float16-64-aliased-ub"),
        pytest.param("float16", 128, True, id="float16-128-scalar-parameter"),
        pytest.param("bfloat16", 128, False, id="bfloat16-128-aliased-ub"),
        pytest.param("bfloat16", 256, True, id="bfloat16-256-scalar-parameter"),
    ],
)
def test_fill_nested_float_casts_preserve_intermediate_rounding(npu, intermediate, lanes, parameter):
    @T.prim_func
    def main(A: T.Tensor((256,), "float32"), value: T.float32, B: T.Tensor((2, 256), "float32")):
        with T.Kernel(1):
            b = T.alloc_shared((256,), "float32")
            T.copy(A, b)
            with T.SimdVF(lanes=lanes):
                if parameter:
                    T.fill(b, T.Cast("float32", T.Cast(intermediate, value)))
                else:
                    T.fill(b, T.Cast("float32", T.Cast(intermediate, b[70])))
            T.copy(b, B[0, :])
            with T.SimdVF(lanes=lanes):
                T.clear(b)
                T.fill(b, T.Cast("float32", T.Cast(intermediate, b[70])))
            T.copy(b, B[1, :])

    kernel = tilelang.compile(main, target="pto", out_idx=-1)
    for scalar in [1.0003, -7.003]:
        initial = torch.arange(256).float() / 13
        initial[70] = scalar
        rounded = torch.tensor(scalar).to(getattr(torch, intermediate)).float().item()
        assert rounded != initial[70].item(), "The input must expose an incorrectly eliminated narrowing cast"
        expected = torch.zeros((2, 256))
        expected[0] = rounded
        got = kernel(initial.to(npu), scalar)
        torch.npu.synchronize()
        torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)


@pytest.mark.pto
@pytest.mark.parametrize("unroll", [False, True], ids=["serial", "unrolled"])
def test_fill_serial_value_with_full_buffer(npu, unroll):
    @T.prim_func
    def main(A: T.Tensor((3,), "float32"), B: T.Tensor((3, 128), "float32")):
        with T.Kernel(1):
            values = T.alloc_shared((3,), "float32")
            b = T.alloc_shared((3, 128), "float32")
            T.copy(A, values)
            with T.SimdVF(lanes=128):
                if unroll:
                    for row in T.unroll(3):
                        T.fill(b, values[row])
                else:
                    for row in T.serial(3):
                        T.fill(b, values[row])
            T.copy(b, B)

    initial = torch.tensor([1.25, -7.5, 3.0])
    kernel = tilelang.compile(main, target="pto", out_idx=-1)
    got = kernel(initial.to(npu))
    torch.npu.synchronize()
    torch.testing.assert_close(got.cpu(), initial[-1].expand(3, 128), rtol=0, atol=0)


@pytest.mark.pto
@pytest.mark.parametrize("shape", [(128,), (1, 128)], ids=["one-dimensional", "logical-unit-axis"])
@pytest.mark.parametrize("versions", [1, 2])
@pytest.mark.parametrize("clear", [False, True], ids=["fill", "clear"])
def test_fill_clear_multibuffer_keeps_real_logical_axes(npu, shape, versions, clear):
    @T.prim_func
    def main(A: T.Tensor((4, *shape), "float32"), B: T.Tensor((4, *shape), "float32")):
        with T.Kernel(1):
            b = T.alloc_shared(shape, "float32")
            T.annotate_buffer_versions({b: versions})
            for t in T.Pipelined(4, num_stages=2):
                if len(shape) == 1:
                    T.copy(A[t, :], b)
                else:
                    T.copy(A[t, :, :], b)
                with T.SimdVF(lanes=128):
                    if clear:
                        T.clear(b)
                    else:
                        if len(shape) == 1:
                            T.fill(b, b[0])
                        else:
                            T.fill(b, b[0, 0])
                if len(shape) == 1:
                    T.copy(b, B[t, :])
                else:
                    T.copy(b, B[t, :, :])

    initial = torch.arange(4 * 128).reshape(4, *shape).float() / 8
    expected = torch.zeros_like(initial)
    if not clear:
        expected = initial.reshape(4, -1)[:, 0:1].expand(4, 128).reshape_as(initial)
    kernel = tilelang.compile(main, target="pto", out_idx=-1)
    got = kernel(initial.to(npu))
    torch.npu.synchronize()
    torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)


@pytest.mark.pto
@pytest.mark.parametrize("shape, lanes", [((128, 1), 128), ((256, 1), 64), ((1, 256), 256)])
@pytest.mark.parametrize("versions", [1, 2])
@pytest.mark.parametrize("clear", [False, True], ids=["fill", "clear"])
def test_fill_clear_compact_singleton_axis(npu, shape, lanes, versions, clear):
    @T.prim_func
    def main(A: T.Tensor((4, *shape), "float32"), B: T.Tensor((4, *shape), "float32")):
        with T.Kernel(1):
            b = T.alloc_shared(shape, "float32")
            if versions > 1:
                T.annotate_buffer_versions({b: versions})
            for t in T.Pipelined(4, num_stages=2):
                T.copy(A[t, :, :], b)
                with T.SimdVF(lanes=lanes):
                    if clear:
                        T.clear(b)
                    else:
                        T.fill(b, b[0, 0])
                T.copy(b, B[t, :, :])

    initial = torch.arange(4 * shape[0] * shape[1]).reshape(4, *shape).float() / 8
    expected = torch.zeros_like(initial)
    if not clear:
        expected[:] = initial[:, :1, :1]
    kernel = tilelang.compile(main, target="pto", out_idx=-1)
    got = kernel(initial.to(npu))
    torch.npu.synchronize()
    torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)


@pytest.mark.pto
def test_fill_reduce_and_parallel_share_source_order(npu):
    @T.prim_func
    def main(B: T.Tensor((128,), "float32")):
        with T.Kernel(1):
            source = T.alloc_shared((3, 128), "float32")
            reduced = T.alloc_shared((128,), "float32")
            b = T.alloc_shared((128,), "float32")
            with T.SimdVF(lanes=128):
                T.fill(source, T.float32(2))
                T.fill(reduced, T.float32(-5))
                T.reduce_sum(source, reduced, dim=0, clear=False)
                T.fill(b, reduced[70])
                T.clear(source)
                for i in T.Parallel(128):
                    b[i] = b[i] + reduced[i] + source[0, i]
            T.copy(b, B)

    kernel = tilelang.compile(main, target="pto", out_idx=-1)
    got = kernel()
    torch.npu.synchronize()
    # Each reduction yields -5 + (2 + 2 + 2) = 1. Both the scalar fill
    # consumer and the elementwise consumer must see that result.
    torch.testing.assert_close(got.cpu(), torch.full((128,), 2.0), rtol=0, atol=0)
