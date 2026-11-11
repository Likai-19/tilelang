"""Frontend kernels for metadata in automatic indexed memory accesses."""

import pytest

import tilelang.ascend.language as T
from test_pto_parallel_gather_scatter import calls, lower_case


def index_view_stride_kernel(kind="constant"):
    @T.prim_func
    def main(A: T.Tensor((256,), "float32"), I: T.Tensor((256,), "int32"), C: T.Tensor((128,), "float32")):
        with T.Kernel(1):
            x = T.alloc_shared((256,), "float32")
            idx = T.alloc_shared((256,), "int32")
            y = T.alloc_shared((128,), "float32")
            T.copy(A, x)
            T.copy(I, idx)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(128):
                    if kind == "indexed":
                        view = T.view(idx, shape=(2, 128), strides=(2 * i, 1))
                    elif kind == "unused":
                        view = T.view(idx, shape=(2, 128), strides=(i, 1))
                    else:
                        view = T.view(idx, shape=(2, 128), strides=(128, 1))
                    if kind == "unused":
                        y[i] = x[view[0, 0]]
                    else:
                        y[i] = x[view[1, 0]]
            T.copy(y, C)

    return main


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["constant", "unused"])
def test_uniform_index_view_only_uses_participating_metadata(kind):
    _, captured = lower_case(f"index_view_stride_{kind}", index_view_stride_kernel(kind))
    converted = captured["VectorizeParallelToPTO_after"]
    assert not calls(converted, "tl.vmi.vgather")
    assert len(calls(converted, "tl.vmi.vbrc")) == 1


@pytest.mark.pto
def test_indexed_view_metadata_preserves_varying_nested_reads():
    _, captured = lower_case("indexed_view_stride", index_view_stride_kernel("indexed"))
    converted = captured["VectorizeParallelToPTO_after"]
    assert len(calls(converted, "tl.vmi.vgather")) == 2


def payload_view_stride_kernel(kind="gather", dtype="float32", owner_size=512, *, loaded_index=True):
    source_size = owner_size if kind == "gather" else 128
    output_size = 128 if kind == "gather" else owner_size
    index_dtype = "uint16" if dtype == "float16" else "int32"

    @T.prim_func
    def main(A: T.Tensor((source_size,), dtype), I: T.Tensor((128,), index_dtype), C: T.Tensor((output_size,), dtype)):
        with T.Kernel(1):
            x = T.alloc_shared((source_size,), dtype)
            idx = T.alloc_shared((128,), index_dtype)
            y = T.alloc_shared((output_size,), dtype)
            T.copy(A, x)
            T.copy(I, idx)
            T.copy(C, y)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(128):
                    if kind == "gather":
                        view = T.view(x, shape=(2, owner_size // 2), strides=(2 * i, 1))
                        if loaded_index:
                            y[i] = view[1, idx[i]]
                        else:
                            y[i] = view[1, 0]
                    else:
                        view = T.view(y, shape=(2, owner_size // 2), strides=(2 * i, 1))
                        if loaded_index:
                            view[1, idx[i]] = x[i]
                        else:
                            view[1, 0] = x[i]
            T.copy(y, C)

    return main


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
def test_indexed_payload_view_retains_lane_varying_stride(kind):
    _, captured = lower_case(f"payload_stride_{kind}", payload_view_stride_kernel(kind))
    converted = captured["VectorizeParallelToPTO_after"]
    indexed = calls(converted, "tl.vmi.vgather" if kind == "gather" else "tl.vmi.vscatter")
    assert len(indexed) == 1
    assert len(calls(converted, "tl.vmi.vmul")) == 1
    assert len(calls(converted, "tl.vmi.vadd")) == 1


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
def test_b16_rebase_cannot_remove_lane_varying_payload_stride(kind):
    with pytest.raises(Exception, match="ui16 relative window"):
        lower_case(f"rejected_b16_payload_stride_{kind}", payload_view_stride_kernel(kind, "float16", 70000))


@T.prim_func
def continuous_payload_view_stride(A: T.Tensor((512,), "float32"), C: T.Tensor((256,), "float32")):
    with T.Kernel(1):
        x = T.alloc_shared((512,), "float32")
        y = T.alloc_shared((256,), "float32")
        T.copy(A, x)
        with T.SimdVF(lanes=128):
            for i in T.Parallel(256):
                view = T.view(x, shape=(2, 256), strides=(i, 1))
                y[i] = view[1, 0]
        T.copy(y, C)


@pytest.mark.pto
def test_continuous_payload_stride_does_not_leave_original_parallel_variable():
    artifact, captured = lower_case("continuous_payload_stride", continuous_payload_view_stride)
    assert len(calls(captured["VectorizeParallelToPTO_after"], "tl.vmi.vload")) == 1
    assert "pto.vmi.vload(" in artifact.kernel_source


def unused_loaded_stride_kernel(uniform=False):
    @T.prim_func
    def main(A: T.Tensor((256,), "float32"), I: T.Tensor((256,), "int32"), C: T.Tensor((256,), "float32")):
        with T.Kernel(1):
            x = T.alloc_shared((256,), "float32")
            idx = T.alloc_shared((256,), "int32")
            y = T.alloc_shared((256,), "float32")
            T.copy(A, x)
            T.copy(I, idx)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(256):
                    old = idx[i]
                    idx[i] = 0
                    view = T.view(x, shape=(1, 256), strides=(old, 1))
                    if uniform:
                        y[i] = view[0, 0]
                    else:
                        y[i] = view[0, i]
            T.copy(y, C)

    return main


@pytest.mark.pto
@pytest.mark.parametrize("uniform", [False, True])
def test_unused_view_stride_does_not_require_a_scalar_loaded_value(uniform):
    _, captured = lower_case(f"unused_loaded_stride_{uniform}", unused_loaded_stride_kernel(uniform))
    converted = captured["VectorizeParallelToPTO_after"]
    assert not calls(converted, "tl.vmi.vgather")
    assert len(calls(converted, "tl.vmi.vload")) == int(not uniform)
