"""B16 windows within allocations larger than ui16 offsets."""

import pytest

from test_pto_parallel_gather_scatter import T, calls, lower_case, tvm


@T.prim_func
def outer_range_window_kernel(A: T.Tensor((70000,), "float16"), I: T.Tensor((128,), "uint16"), C: T.Tensor((2, 128), "float16")):
    with T.Kernel(2) as bx:
        x = T.alloc_shared((70000,), "float16")
        idx = T.alloc_shared((128,), "uint16")
        y = T.alloc_shared((128,), "float16")
        view = T.view(x, shape=(2, 35000))
        T.copy(A, x)
        T.copy(I, idx)
        with T.SimdVF(lanes=128):
            for i in T.Parallel(128):
                y[i] = view[bx, idx[i]]
        T.copy(y, C[bx, :])


@pytest.mark.pto
def test_b16_base_uses_lexical_launch_bounds_only_during_verification():
    _, captured = lower_case("outer_range_window", outer_range_window_kernel)
    converted = captured["VectorizeParallelToPTO_after"]
    gather = calls(converted, "tl.vmi.vgather")[0]
    assert str(gather.args[2].dtype) == "uint16x128"
    assert "35000" in str(gather.args[1])


def alternate_axis_window_kernel(kind="gather"):
    source_shape = (128, 600) if kind == "gather" else (2, 128)
    output_shape = (2, 128) if kind == "gather" else (128, 600)

    @T.prim_func
    def main(A: T.Tensor(source_shape, "float16"), C: T.Tensor(output_shape, "float16")):
        with T.Kernel(1):
            x = T.alloc_shared(source_shape, "float16")
            y = T.alloc_shared(output_shape, "float16")
            T.copy(A, x)
            T.copy(C, y)
            with T.SimdVF(lanes=128):
                for i, j in T.Parallel(2, 128):
                    if kind == "gather":
                        y[i, j] = x[j, 2 * i]
                    else:
                        y[j, 2 * i] = x[i, j]
            T.copy(y, C)

    return main


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
def test_b16_affine_offsets_reject_unrepresentable_window_or_implicit_narrowing(kind):
    # Neither axis is lowerable: the wide direction exceeds the ui16 window,
    # while the other still requires forbidden implicit int32-to-ui16 narrowing.
    with pytest.raises(Exception, match="implicit narrowing|ui16 relative window"):
        lower_case(f"rejected_alternate_axis_{kind}", alternate_axis_window_kernel(kind))


def base_outside_owner_kernel():
    @T.prim_func
    def main(A: T.Tensor((70000,), "float16"), I: T.Tensor((128,), "uint16"), C: T.Tensor((128,), "float16")):
        with T.Kernel(1):
            raw = T.alloc_shared((70000,), "float16")
            # The view's uniform origin sits exactly at the owner's end: its
            # relative window is representable, but the rebased scalar base
            # is provably outside the owner, and the diagnostic must name the
            # base contract instead of blaming the ui16 window.
            view = tvm.tirx.decl_buffer((256,), "float16", data=raw.data, elem_offset=70000, scope="shared.dyn")
            idx = T.alloc_shared((128,), "uint16")
            y = T.alloc_shared((128,), "float16")
            T.copy(A, raw)
            T.copy(I, idx)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(128):
                    y[i] = view[idx[i]]
            T.copy(y, C)

    return main


@pytest.mark.pto
def test_rebased_base_outside_owner_reports_the_base_contract():
    with pytest.raises(Exception, match="scalar instruction base D.*inside the owner.*70000"):
        lower_case("rejected_base_outside_owner", base_outside_owner_kernel())
