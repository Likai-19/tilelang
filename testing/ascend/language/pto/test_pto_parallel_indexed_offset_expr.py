"""Verify complete indexed offsets using the expression lowering contract."""

import pytest

from test_pto_parallel_gather_scatter import ascend_transform, calls, indexed_kernel, lower_case, tvm
from test_pto_parallel_indexed_control_flow import replace_unit_body
from tilelang.ascend.target import normalize_pto_target
import tilelang.ascend.language as T


def scalar_select_coordinate_cast_kernel(extent=256):
    @T.prim_func
    def main(A: T.Tensor((512,), "float32"), I: T.Tensor((256,), "int32"), C: T.Tensor((256,), "float32")):
        with T.Kernel(1):
            x = T.alloc_shared((512,), "float32")
            idx = T.alloc_shared((256,), "int32")
            y = T.alloc_shared((256,), "float32")
            T.copy(A, x)
            T.copy(I, idx)
            T.copy(C, y)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(extent):
                    y[i] = x[T.Cast("uint32", idx[i] + T.Select(idx[0] > 0, 0, 1))]
            T.copy(y, C)

    return main


@pytest.fixture(scope="module")
def source_module():
    _, captured = lower_case("indexed_offset_expression_base", indexed_kernel())
    return captured["VerifyParallelToPTO_before"]


@pytest.mark.pto
def test_cast_input_may_include_a_scalar_select_coordinate(source_module):
    def make_body(i, payload, index, output):
        scalar = tvm.tirx.Select(index[0] > 0, 0, 1)
        offset = tvm.tirx.Cast("uint32", index[i] + scalar)
        return tvm.tirx.BufferStore(output, payload[offset], [i])

    mod = replace_unit_body(source_module, make_body)
    with normalize_pto_target("pto"):
        verified = ascend_transform.VerifyParallelToPTO()(mod)
        assert tvm.ir.structural_equal(mod, verified), "verification must not rewrite scalar coordinates or source casts"
        converted = ascend_transform.VectorizeParallelToPTO()(verified)
    assert len(calls(converted, "tl.vmi.vgather")) == 1
    assert len(calls(converted, "tl.vmi.vload")) >= 1
    assert len(calls(converted, "tl.vmi.vinterpret_cast")) == 1, "the explicit same-width source Cast must survive"


@pytest.mark.pto
def test_complete_offset_checks_references_introduced_by_view_metadata(source_module):
    def make_body(i, payload, index, output):
        ghost = tvm.tirx.Var("undefined_view_origin", "int32")
        view = tvm.tirx.decl_buffer(payload.shape, payload.dtype, data=payload.data, elem_offset=ghost, scope=payload.scope())
        return tvm.tirx.BufferStore(output, view[index[i]], [i])

    mod = replace_unit_body(source_module, make_body)
    with normalize_pto_target("pto"), pytest.raises(Exception, match="undefined_view_origin.*never defined"):
        ascend_transform.VerifyParallelToPTO()(mod)


@pytest.mark.pto
def test_uniform_store_checks_operations_introduced_by_view_metadata(source_module):
    def make_body(i, payload, index, output):
        origin = tvm.tirx.call_extern("int32", "opaque_origin")
        view = tvm.tirx.decl_buffer(output.shape, output.dtype, data=output.data, elem_offset=origin, scope=output.scope())
        return tvm.tirx.BufferStore(view, payload[i], [0])

    mod = replace_unit_body(source_module, make_body)
    with normalize_pto_target("pto"), pytest.raises(Exception, match="complete indexed offset.*unsupported call"):
        ascend_transform.VerifyParallelToPTO()(mod)


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["predicate", "conditional_gather"])
def test_metadata_index_reads_keep_their_own_admission_rules(source_module, kind):
    def make_body(i, payload, index, output):
        if kind == "predicate":
            origin = tvm.tirx.BufferLoad(index, [i], predicate=i < 150)
        else:
            origin = tvm.tirx.Select(index[0] > 0, index[index[i]], index[i])
        view = tvm.tirx.decl_buffer(payload.shape, payload.dtype, data=payload.data, elem_offset=origin, scope=payload.scope())
        if kind == "conditional_gather":
            return tvm.tirx.Evaluate(view[index[i]])
        return tvm.tirx.BufferStore(output, view[index[i]], [i])

    mod = replace_unit_body(source_module, make_body)
    diagnostic = (
        "non-trivial access predicate"
        if kind == "predicate"
        else "requires gather.*Select true value|unsupported indexed offset node.*Select"
    )
    with normalize_pto_target("pto"), pytest.raises(Exception, match=diagnostic):
        ascend_transform.VerifyParallelToPTO()(mod)


@T.prim_func
def nested_gather_kernel(A: T.Tensor((256,), "float32"), I: T.Tensor((256,), "int32"), C: T.Tensor((256,), "float32")):
    with T.Kernel(1):
        x = T.alloc_shared((256,), "float32")
        idx = T.alloc_shared((256,), "int32")
        y = T.alloc_shared((256,), "float32")
        T.copy(A, x)
        T.copy(I, idx)
        with T.SimdVF(lanes=128):
            for i in T.Parallel(256):
                y[i] = x[idx[idx[idx[idx[idx[idx[idx[idx[idx[idx[idx[idx[idx[idx[i]]]]]]]]]]]]]]]
        T.copy(y, C)


@pytest.mark.pto
def test_nested_gather_offsets_preserve_each_read_once():
    _, captured = lower_case("nested_gather_full_pipeline", nested_gather_kernel)
    converted = captured["VectorizeParallelToPTO_after"]
    assert len(calls(converted, "tl.vmi.vload")) == 1
    assert len(calls(converted, "tl.vmi.vgather")) == 14


@pytest.mark.pto
def test_bound_external_view_origin_survives_output_reference_check(source_module):
    base = tvm.tirx.Var("external_view_origin", "int32")

    def make_body(i, payload, index, output):
        view = tvm.tirx.decl_buffer(payload.shape, payload.dtype, data=payload.data, elem_offset=base, scope=payload.scope())
        return tvm.tirx.BufferStore(output, view[index[i]], [i])

    mod = replace_unit_body(source_module, make_body)
    functions = {}
    for name, func in mod.functions.items():
        functions[name] = tvm.tirx.PrimFunc([*func.params, base], func.body, func.ret_type, func.buffer_map, func.attrs)
    mod = tvm.IRModule(functions)
    with normalize_pto_target("pto"):
        verified = ascend_transform.VerifyParallelToPTO()(mod)
        converted = ascend_transform.VectorizeParallelToPTO()(verified)
    assert len(calls(converted, "tl.vmi.vgather")) == 1


@T.prim_func
def nested_b16_rebased_gather_kernel(A: T.Tensor((70000,), "uint16"), I: T.Tensor((128,), "uint16"), C: T.Tensor((128,), "uint16")):
    with T.Kernel(1):
        raw = T.alloc_shared((70000,), "uint16")
        view = tvm.tirx.decl_buffer((256,), "uint16", data=raw.data, elem_offset=66000, scope="shared.dyn")
        idx = T.alloc_shared((128,), "uint16")
        y = T.alloc_shared((128,), "uint16")
        T.copy(A, raw)
        T.copy(I, idx)
        with T.SimdVF(lanes=128):
            for i in T.Parallel(128):
                y[i] = view[view[view[view[view[view[view[view[view[view[view[view[view[view[idx[i]]]]]]]]]]]]]]]
        T.copy(y, C)


@pytest.mark.pto
def test_nested_b16_gathers_verify_metadata_without_rewalking_source_indices():
    _, captured = lower_case("nested_b16_rebased_gather", nested_b16_rebased_gather_kernel)
    converted = captured["VectorizeParallelToPTO_after"]
    gathered = calls(converted, "tl.vmi.vgather")
    assert len(gathered) == 14
    assert all(int(call.args[1]) == 66000 for call in gathered)
    assert len(calls(converted, "tl.vmi.vload")) == 1


# FloorDiv/FloorMod in indexed offsets are not adapted: any form that
# depends on the vectorized lane coordinate has no gather/scatter lowering
# and is rejected directly. Only pure scalar coordinates independent of the
# unit's values (launch variables and compiler-generated buffer-version
# coordinates such as `t % stages`) are preserved as scalars; see the
# `versioned` design case in test_pto_parallel_gather_scatter.py.


def varying_floor_offset_kernel(kind):
    @T.prim_func
    def main(A: T.Tensor((256,), "float32"), I: T.Tensor((256,), "int32"), C: T.Tensor((256,), "float32")):
        with T.Kernel(1):
            x = T.alloc_shared((256,), "float32")
            idx = T.alloc_shared((256,), "int32")
            y = T.alloc_shared((256,), "float32")
            T.copy(A, x)
            T.copy(I, idx)
            T.copy(C, y)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(256):
                    if kind == "gather_floordiv":
                        y[i] = x[i // 2]
                    elif kind == "gather_floormod":
                        y[i] = x[idx[i] + i % 3]
                    elif kind == "gather_chunk_floordiv":
                        view = T.view(x, shape=(2, 128), strides=(128, 1))
                        y[i] = view[i // 128, idx[i]]
                    elif kind == "scatter_floordiv":
                        y[i // 2] = x[i]
                    else:
                        assert kind == "stride_floormod"
                        view = T.view(x, shape=(2, 128), strides=(i % 3, 1))
                        y[i] = view[1, 0]
            T.copy(y, C)

    return main


@pytest.mark.pto
@pytest.mark.parametrize(
    "kind",
    ["gather_floordiv", "gather_floormod", "gather_chunk_floordiv", "stride_floormod"],
)
def test_varying_floor_offsets_are_rejected_without_adaptation(kind):
    # The failing address belongs to a load nested in the store value, so the
    # diagnostic must name that buffer instead of blaming the value itself.
    with pytest.raises(
        Exception,
        match=r"nested access to buffer .+: unsupported indexed offset node tirx.Floor(Div|Mod)",
    ):
        lower_case(f"rejected_floor_{kind}", varying_floor_offset_kernel(kind))


@pytest.mark.pto
def test_varying_floor_scatter_target_is_rejected_with_buffer_context():
    # A store whose own index fails reports through the memory-access
    # diagnostic; the buffer context comes from the reporting site.
    with pytest.raises(
        Exception,
        match=r"memory access: buffer `y`: unsupported indexed offset node tirx.FloorDiv",
    ):
        lower_case("rejected_floor_scatter_floordiv", varying_floor_offset_kernel("scatter_floordiv"))


def scalar_floor_coordinate_kernel(kind):
    @T.prim_func
    def main(A: T.Tensor((256,), "float32"), I: T.Tensor((128,), "int32"), C: T.Tensor((4, 256), "float32")):
        with T.Kernel(4) as bx:
            x = T.alloc_shared((256,), "float32")
            idx = T.alloc_shared((128,), "int32")
            y = T.alloc_shared((256,), "float32")
            T.copy(A, x)
            T.copy(I, idx)
            T.copy(C[bx, :], y)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(128):
                    if kind == "gather":
                        y[i] = x[idx[i] + bx % 2]
                    else:
                        y[idx[i] + bx // 2] = x[i]
            T.copy(y, C[bx, :])

    return main


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
def test_scalar_floor_coordinates_stay_scalar_in_indexed_offsets(kind):
    # `bx % 2` / `bx // 2` do not depend on the lane, so they are preserved as
    # scalar coordinates and enter the instruction once as a broadcast —
    # the same treatment compiler-generated buffer-version coordinates
    # receive. Simplification may rewrite the operator (e.g. to a bitwise
    # form), so the contract checked here is the scalar dtype of the single
    # broadcast, not the surviving node type.
    _, captured = lower_case(f"scalar_floor_{kind}", scalar_floor_coordinate_kernel(kind))
    converted = captured["VectorizeParallelToPTO_after"]
    operation = "tl.vmi.vgather" if kind == "gather" else "tl.vmi.vscatter"
    assert len(calls(converted, operation)) == 1
    assert not calls(converted, "tl.vmi.vcvt")
    broadcasts = calls(converted, "tl.vmi.vbrc")
    assert len(broadcasts) == 1, "the scalar floor coordinate is broadcast once"
    assert str(broadcasts[0].args[0].dtype) == "int32", "the floor coordinate stays a scalar"
