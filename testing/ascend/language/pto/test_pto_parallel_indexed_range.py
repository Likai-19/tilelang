"""Range-proven B16 offsets are generated directly in uint16 arithmetic."""

import pytest

import tilelang.ascend.language as T
from test_pto_parallel_gather_scatter import assert_vector_axis, calls, lower_case


def b16_transposed_index_kernel(force_scatter=False):
    output_shape = (2, 128, 128) if force_scatter else (128, 128)

    @T.prim_func
    def main(A: T.Tensor((256,), "float16"), I: T.Tensor((128,), "uint16"), C: T.Tensor(output_shape, "float16")):
        with T.Kernel(1):
            x = T.alloc_shared((256,), "float16")
            idx = T.alloc_shared((128,), "uint16")
            y = T.alloc_shared(output_shape, "float16")
            T.copy(A, x)
            T.copy(I, idx)
            with T.SimdVF(lanes=128):
                for i, j in T.Parallel(128, 128):
                    if force_scatter:
                        y[0, j, i] = x[idx[i]]
                        y[1, i, j] = x[j]
                    else:
                        y[j, i] = x[idx[i]]
            T.copy(y, C)

    return main


@pytest.mark.pto
@pytest.mark.parametrize("force_scatter", [False, True])
def test_b16_transposed_index_uses_proven_offsets_without_vector_conversion(force_scatter):
    artifact, captured = lower_case(f"b16_transposed_index_{force_scatter}", b16_transposed_index_kernel(force_scatter))
    converted = captured["VectorizeParallelToPTO_after"]
    assert not calls(converted, "tl.vmi.vcvt")
    assert "pto.vmi.vcvt(" not in artifact.kernel_source
    if force_scatter:
        assert_vector_axis(captured["VerifyParallelToPTO_before"], 1)
        scatter = calls(converted, "tl.vmi.vscatter")
        assert len(scatter) == 1 and str(scatter[0].args[3].dtype) == "uint16x128"
        coordinates = calls(converted, "tl.vmi.vci")
        assert coordinates and all(str(call.dtype) == "uint16x128" for call in coordinates)
        arithmetic = calls(converted, "tl.vmi.vmul") + calls(converted, "tl.vmi.vadd")
        assert arithmetic and all(str(call.dtype) == "uint16x128" for call in arithmetic)


def affine_b16_scatter_kernel(extent=150, stride=2, origin=1, output_size=512):
    @T.prim_func
    def main(A: T.Tensor((256,), "float16"), C: T.Tensor((output_size,), "float16")):
        with T.Kernel(1):
            x = T.alloc_shared((256,), "float16")
            y = T.alloc_shared((output_size,), "float16")
            T.copy(A, x)
            T.copy(C, y)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(extent):
                    y[stride * i + origin] = x[i]
            T.copy(y, C)

    return main


@pytest.mark.pto
@pytest.mark.parametrize(
    "extent,stride,origin,output_size",
    [(150, 2, 1, 512), (128, 512, 511, 65536), (127, 516, 519, 65536), (255, 258, 3, 65536), (128, 2, 0, 70000)],
)
def test_affine_b16_offsets_use_uint16_from_the_coordinate_seed(extent, stride, origin, output_size):
    artifact, captured = lower_case("affine_b16_range", affine_b16_scatter_kernel(extent, stride, origin, output_size))
    converted = captured["VectorizeParallelToPTO_after"]
    scatter = calls(converted, "tl.vmi.vscatter")
    assert len(scatter) == 1 and str(scatter[0].args[3].dtype) == "uint16x128"
    assert all(str(call.dtype) == "uint16x128" for call in calls(converted, "tl.vmi.vci"))
    assert not calls(converted, "tl.vmi.vcvt")
    assert "pto.vmi.vcvt(" not in artifact.kernel_source
    if extent % 128:
        assert len(calls(converted, "tl.vmi.vsel")) == 1, "inactive tail offsets must still be sanitized"


@pytest.mark.pto
def test_affine_b16_offset_above_uint16_range_is_rejected():
    # The last active offset is 65536, one beyond the unsigned 16-bit limit.
    with pytest.raises(Exception, match="ui16 relative window|implicit narrowing|representable"):
        lower_case("affine_b16_out_of_range", affine_b16_scatter_kernel(128, 512, 512, 65544))


@T.prim_func
def block_coordinate_b16_scatter(A: T.Tensor((256,), "float16"), C: T.Tensor((2, 512), "float16")):
    with T.Kernel(2) as bx:
        x = T.alloc_shared((256,), "float16")
        y = T.alloc_shared((512,), "float16")
        T.copy(A, x)
        T.copy(C[bx, :], y)
        with T.SimdVF(lanes=128):
            for i in T.Parallel(128):
                y[2 * i + bx] = x[i]
        T.copy(y, C[bx, :])


@pytest.mark.pto
def test_b16_offsets_use_proven_launch_coordinate_range():
    _, captured = lower_case("block_coordinate_b16_scatter", block_coordinate_b16_scatter)
    converted = captured["VectorizeParallelToPTO_after"]
    scatter = calls(converted, "tl.vmi.vscatter")
    assert len(scatter) == 1 and str(scatter[0].args[3].dtype) == "uint16x128"
    assert not calls(converted, "tl.vmi.vcvt")
