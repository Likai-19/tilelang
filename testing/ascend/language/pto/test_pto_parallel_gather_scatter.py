"""Automatic Parallel gather/scatter kernels, IR contracts, and diagnostics."""

from contextlib import contextmanager
import os
from pathlib import Path

import pytest

import tilelang
import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.ascend import transform as ascend_transform
from tilelang.ascend.target import normalize_pto_target
from tilelang.engine.lower import lower


@T.prim_func
def gather_f32(A: T.Tensor((256,), "float32"), I: T.Tensor((256,), "int32"), C: T.Tensor((256,), "float32")):
    with T.Kernel(1):
        a = T.alloc_shared((256,), "float32")
        idx = T.alloc_shared((256,), "int32")
        y = T.alloc_shared((256,), "float32")
        T.copy(A, a)
        T.copy(I, idx)
        with T.SimdVF(lanes=128):
            for i in T.Parallel(256):
                y[i] = a[idx[i]]
        T.copy(y, C)


@T.prim_func
def gather_f16_tail(A: T.Tensor((256,), "float16"), I: T.Tensor((256,), "int16"), C: T.Tensor((256,), "float16")):
    with T.Kernel(1):
        table = T.alloc_shared((256,), "float16")
        idx = T.alloc_shared((256,), "int16")
        y = T.alloc_shared((256,), "float16")
        T.copy(A, table)
        T.copy(I, idx)
        T.copy(C, y)
        with T.SimdVF(lanes=128):
            for i in T.Parallel(150):
                y[i] = table[idx[i]]
        T.copy(y, C)


@T.prim_func
def scatter_f32(A: T.Tensor((256,), "float32"), I: T.Tensor((256,), "int32"), C: T.Tensor((256,), "float32")):
    with T.Kernel(1):
        x = T.alloc_shared((256,), "float32")
        idx = T.alloc_shared((256,), "int32")
        dst = T.alloc_shared((256,), "float32")
        T.copy(A, x)
        T.copy(I, idx)
        T.copy(C, dst)
        with T.SimdVF(lanes=128):
            for i in T.Parallel(256):
                dst[idx[i]] = x[i]
        T.copy(dst, C)


@T.prim_func
def stride2(A: T.Tensor((256,), "float32"), C: T.Tensor((512,), "float32")):
    with T.Kernel(1):
        x = T.alloc_shared((256,), "float32")
        dst = T.alloc_shared((512,), "float32")
        T.copy(A, x)
        T.copy(C, dst)
        with T.SimdVF(lanes=128):
            for i in T.Parallel(256):
                dst[2 * i + 1] = x[i]
        T.copy(dst, C)


@T.prim_func
def saved_index(A: T.Tensor((256,), "float32"), I: T.Tensor((256,), "int32"), C: T.Tensor((256,), "float32")):
    with T.Kernel(1):
        table = T.alloc_shared((256,), "float32")
        idx = T.alloc_shared((256,), "int32")
        y = T.alloc_shared((256,), "float32")
        T.copy(A, table)
        T.copy(I, idx)
        with T.SimdVF(lanes=128):
            for i in T.Parallel(256):
                old = idx[i]
                idx[i] = i + 1
                y[i] = table[old]
        T.copy(y, C)


@T.prim_func
def dynamic_row(A: T.Tensor((4, 256), "float32"), I: T.Tensor((4,), "int32"), C: T.Tensor((4, 256), "float32")):
    with T.Kernel(1):
        x = T.alloc_shared((4, 256), "float32")
        row = T.alloc_shared((4,), "int32")
        y = T.alloc_shared((4, 256), "float32")
        T.copy(A, x)
        T.copy(I, row)
        with T.SimdVF(lanes=128):
            for i, j in T.Parallel(4, 256):
                y[i, j] = x[row[i], j]
        T.copy(y, C)


@T.prim_func
def column_gather(A: T.Tensor((4, 256), "float32"), I: T.Tensor((256,), "int32"), C: T.Tensor((4, 256), "float32")):
    with T.Kernel(1):
        x = T.alloc_shared((4, 256), "float32")
        col = T.alloc_shared((256,), "int32")
        y = T.alloc_shared((4, 256), "float32")
        T.copy(A, x)
        T.copy(I, col)
        with T.SimdVF(lanes=128):
            for i, j in T.Parallel(4, 256):
                y[i, j] = x[i, col[j]]
        T.copy(y, C)


@T.prim_func
def strided_view(A: T.Tensor((800,), "float32"), I: T.Tensor((256,), "int32"), C: T.Tensor((256,), "float32")):
    with T.Kernel(1):
        raw = T.alloc_shared((800,), "float32")
        view = tvm.tirx.decl_buffer((2, 256), "float32", data=raw.data, strides=(512, 1), elem_offset=32, scope="shared.dyn")
        idx = T.alloc_shared((256,), "int32")
        y = T.alloc_shared((256,), "float32")
        T.copy(A, raw)
        T.copy(I, idx)
        with T.SimdVF(lanes=128):
            for i in T.Parallel(256):
                y[i] = view[1, idx[i]]
        T.copy(y, C)


@T.prim_func
def versioned(A: T.Tensor((4, 256), "float32"), I: T.Tensor((4, 256), "int32"), C: T.Tensor((4, 256), "float32")):
    with T.Kernel(1):
        x = T.alloc_shared((256,), "float32")
        idx = T.alloc_shared((256,), "int32")
        y = T.alloc_shared((256,), "float32")
        T.annotate_buffer_versions({x: 2, idx: 2, y: 2})
        for t in T.Pipelined(4, num_stages=2):
            T.copy(A[t, :], x)
            T.copy(I[t, :], idx)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(256):
                    y[i] = x[idx[i]]
            T.copy(y, C[t, :])


@T.prim_func
def nested_gather(
    A: T.Tensor((256,), "float32"), I: T.Tensor((256,), "int32"), J: T.Tensor((256,), "int32"), C: T.Tensor((256,), "float32")
):
    with T.Kernel(1):
        table = T.alloc_shared((256,), "float32")
        idx0 = T.alloc_shared((256,), "int32")
        idx1 = T.alloc_shared((256,), "int32")
        y = T.alloc_shared((256,), "float32")
        T.copy(A, table)
        T.copy(I, idx0)
        T.copy(J, idx1)
        with T.SimdVF(lanes=128):
            for i in T.Parallel(256):
                y[i] = table[idx1[idx0[i]]]
        T.copy(y, C)


@T.prim_func
def store_then_gather(A: T.Tensor((256,), "float32"), I: T.Tensor((256,), "int32"), C: T.Tensor((256,), "float32")):
    with T.Kernel(1):
        x = T.alloc_shared((256,), "float32")
        work = T.alloc_shared((256,), "float32")
        alias = tvm.tirx.decl_buffer((256,), "float32", data=work.data, strides=(1,), scope="shared.dyn")
        idx = T.alloc_shared((256,), "int32")
        y = T.alloc_shared((256,), "float32")
        T.copy(A, x)
        T.copy(I, idx)
        with T.SimdVF(lanes=128):
            for i in T.Parallel(256):
                work[i] = x[i]
            for i in T.Parallel(256):
                y[i] = alias[idx[i]]
        T.copy(y, C)


@T.prim_func
def rejected_fragment(A: T.Tensor((128,), "float32"), I: T.Tensor((128,), "int32"), C: T.Tensor((128,), "float32")):
    with T.Kernel(1):
        a = T.alloc_shared((128,), "float32")
        idx = T.alloc_shared((128,), "int32")
        frag = T.alloc_fragment((128,), "float32")
        T.copy(A, a)
        T.copy(I, idx)
        with T.SimdVF(lanes=128):
            for i in T.Parallel(128):
                frag[i] = a[idx[i]]
        T.copy(frag, C)


@T.prim_func
def rejected_wide_window(A: T.Tensor((70000,), "float16"), I: T.Tensor((128,), "int32"), C: T.Tensor((128,), "float16")):
    with T.Kernel(1):
        table = T.alloc_shared((70000,), "float16")
        idx = T.alloc_shared((128,), "int32")
        y = T.alloc_shared((128,), "float16")
        T.copy(A, table)
        T.copy(I, idx)
        with T.SimdVF(lanes=128):
            for i in T.Parallel(128):
                y[i] = table[idx[i]]
        T.copy(y, C)


# Parameterized kernels exercise native dtype widths and tail masks.


def indexed_kernel(kind="gather", dtype="float32", lanes=128, extent=256, *, manual_sync=False, index_dtype=None):
    padded = (extent + lanes - 1) // lanes * lanes
    owner_size = max(256, padded)
    index_dtype = index_dtype or ("int16" if dtype in ("float16", "bfloat16", "int16", "uint16") else "int32")

    @T.prim_func
    def main(A: T.Tensor((owner_size,), dtype), I: T.Tensor((padded,), index_dtype), C: T.Tensor((owner_size,), dtype)):
        with T.Kernel(1):
            x = T.alloc_shared((owner_size,), dtype)
            idx = T.alloc_shared((padded,), index_dtype)
            y = T.alloc_shared((owner_size,), dtype)
            T.copy(A, x)
            T.copy(I, idx)
            T.copy(C, y)
            if manual_sync:
                T.ascend_set_flag("MTE2_V", 0)
                T.ascend_wait_flag("MTE2_V", 0)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(extent):
                    if kind == "gather":
                        y[i] = x[idx[i]]
                    else:
                        y[idx[i]] = x[i]
            if manual_sync:
                T.ascend_set_flag("V_MTE3", 0)
                T.ascend_wait_flag("V_MTE3", 0)
            T.copy(y, C)

    return main


DESIGN_CASES = {
    "G1_gather_f32": gather_f32,
    "G2_gather_f16_tail": gather_f16_tail,
    "S1_scatter_f32": scatter_f32,
    "S3_stride2": stride2,
    "M1_saved_index": saved_index,
    "U2_dynamic_row": dynamic_row,
    "D2_column_gather": column_gather,
    "V1_strided_view": strided_view,
    "V2_versioned": versioned,
    "G3_nested_gather": nested_gather,
    "O1_alias_dependency": store_then_gather,
}


@contextmanager
def capture_passes():
    captured = {}
    originals = {}
    for name in ("VerifyParallelToPTO", "VectorizeParallelToPTO"):
        original = getattr(ascend_transform, name)
        originals[name] = original

        def factory(original=original, name=name):
            transform = original()

            def run(mod):
                captured[name + "_before"] = mod
                result = transform(mod)
                captured[name + "_after"] = result
                return result

            return run

        setattr(ascend_transform, name, factory)
    try:
        yield captured
    finally:
        for name, original in originals.items():
            setattr(ascend_transform, name, original)


def dump_evidence(name, captured, source=None):
    root = os.environ.get("TILELANG_GS_EVIDENCE_DIR")
    if root is None:
        return
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    for stage, mod in captured.items():
        (path / f"{name}.{stage}.tir").write_text(mod.script())
    if source is not None:
        (path / f"{name}.ptodsl.py").write_text(source)


def lower_case(name, func, *, auto_schedule=True):
    target = normalize_pto_target("pto")
    with capture_passes() as captured:
        try:
            with target, tvm.transform.PassContext(config={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: auto_schedule}):
                artifact = lower(func, target=target)
        finally:
            dump_evidence(name, captured)
    dump_evidence(name, captured, artifact.kernel_source)
    return artifact, captured


def calls(mod, op_name):
    result = []
    for func in mod.functions.values():
        tvm.tirx.stmt_functor.post_order_visit(
            func.body,
            lambda node: result.append(node) if isinstance(node, tvm.tirx.Call) and getattr(node.op, "name", "") == op_name else None,
        )
    return result


@T.prim_func
def transpose_2d(A: T.Tensor((4, 256), "float32"), C: T.Tensor((256, 4), "float32")):
    with T.Kernel(1):
        x = T.alloc_shared((4, 256), "float32")
        # Keep UB-to-GM copy row starts 32-byte aligned.
        y = T.alloc_shared((256, 8), "float32")
        T.copy(A, x)
        with T.SimdVF(lanes=128):
            for i, j in T.Parallel(256, 4):
                y[i, j] = x[j, i]
        T.copy(y[:, :4], C)


@T.prim_func
def nested_index_candidate(A: T.Tensor((16384,), "float32"), I: T.Tensor((128, 128), "int32"), C: T.Tensor((128, 128), "float32")):
    with T.Kernel(1):
        x = T.alloc_shared((16384,), "float32")
        idx = T.alloc_shared((128, 128), "int32")
        y = T.alloc_shared((128, 128), "float32")
        T.copy(A, x)
        T.copy(I, idx)
        with T.SimdVF(lanes=64):
            for i, j in T.Parallel(128, 128):
                y[i, j] = x[idx[j, i]]
        T.copy(y, C)


def candidate_ranking_kernel(kind):
    @T.prim_func
    def main(A: T.Tensor((128, 128), "float32"), C: T.Tensor((2, 128, 128), "float32")):
        with T.Kernel(1):
            x = T.alloc_shared((128, 128), "float32")
            y = T.alloc_shared((2, 128, 128), "float32")
            T.copy(A, x)
            T.copy(C, y)
            with T.SimdVF(lanes=64):
                for i, j in T.Parallel(128, 128):
                    if kind == "load_count":
                        y[0, i, j] = x[j, i] + x[i, j]
                        y[1, i, j] = x[j, i]
                    elif kind == "store_count":
                        y[0, j, i] = x[j, i] + x[i, j]
                        y[1, j, i] = 1.0
                    elif kind == "equal_counts":
                        y[0, i, j] = x[j, i] + x[i, j]
                        y[1, j, i] = 1.0
                    elif kind == "scatter_count":
                        y[0, j, i] = x[i, j]
                    elif kind == "ordinary_inner":
                        y[0, i, j] = x[i, j]
                    else:
                        y[0, j, i] = x[j, i]
            T.copy(y, C)

    return main


def degenerate_candidate_kernel(outer, stride):
    rows, columns = (128, 1) if outer else (1, 128)

    @T.prim_func
    def main(C: T.Tensor((256,), "float32")):
        with T.Kernel(1):
            y = T.alloc_shared((256,), "float32")
            T.copy(C, y)
            with T.SimdVF(lanes=64):
                for i, j in T.Parallel(rows, columns):
                    y[stride * (i + j)] = 1.0
            T.copy(y, C)

    return main


def assert_vector_axis(mod, axis):
    layouts = []
    for func in mod.functions.values():
        tvm.tirx.stmt_functor.post_order_visit(
            func.body,
            lambda node: (
                layouts.append(node.annotations["pto_parallel_loop_layout"])
                if isinstance(node, tvm.tirx.For) and "pto_parallel_loop_layout" in node.annotations
                else None
            ),
        )
    assert len(layouts) == 1
    analyzer = tvm.arith.Analyzer()
    assert int(analyzer.simplify(layouts[0].map_forward_thread([1, 0])[0])) == int(axis == 0)
    assert int(analyzer.simplify(layouts[0].map_forward_thread([0, 1])[0])) == int(axis == 1)


@pytest.mark.pto
def test_transpose_prefers_fewer_iterations():
    _, captured = lower_case("transpose_2d", transpose_2d)
    assert_vector_axis(captured["VerifyParallelToPTO_before"], 0)
    after = captured["VectorizeParallelToPTO_after"]
    assert not calls(after, "tl.vmi.vgather")
    assert len(calls(after, "tl.vmi.vload")) == 1
    assert len(calls(after, "tl.vmi.vscatter")) == 1
    loops = {}
    for func in after.functions.values():
        tvm.tirx.stmt_functor.post_order_visit(
            func.body,
            lambda node: loops.update({node.loop_var.name: int(node.extent)}) if isinstance(node, tvm.tirx.For) else None,
        )
    assert loops == {"j": 4, "q": 2}, "i-axis vectorization must execute 8 vector iterations instead of 256"


@pytest.mark.pto
def test_2d_candidate_counts_nested_index_loads():
    _, captured = lower_case("candidate_nested_index", nested_index_candidate)
    assert_vector_axis(captured["VerifyParallelToPTO_before"], 1)
    after = captured["VectorizeParallelToPTO_after"]
    assert len(calls(after, "tl.vmi.vgather")) == 2
    assert not calls(after, "tl.vmi.vscatter")
    assert not calls(after, "tl.vmi.vload")


@pytest.mark.pto
@pytest.mark.parametrize(
    "kind,axis,gathers,scatters",
    [
        ("load_count", 1, 2, 0),
        ("store_count", 0, 1, 0),
        ("equal_counts", 1, 1, 1),
        ("scatter_count", 0, 1, 0),
        ("ordinary_inner", 1, 0, 0),
        ("ordinary_outer", 0, 0, 0),
    ],
)
def test_2d_candidate_ranking(kind, axis, gathers, scatters):
    _, captured = lower_case(f"candidate_{kind}", candidate_ranking_kernel(kind))
    assert_vector_axis(captured["VerifyParallelToPTO_before"], axis)
    after = captured["VectorizeParallelToPTO_after"]
    assert len(calls(after, "tl.vmi.vgather")) == gathers
    assert len(calls(after, "tl.vmi.vscatter")) == scatters


@pytest.mark.pto
@pytest.mark.parametrize("outer", [True, False])
@pytest.mark.parametrize("stride", [1, 2])
def test_equal_indexed_counts_preserve_degenerate_axis_rule(outer, stride):
    _, captured = lower_case(f"candidate_degenerate_{outer}_{stride}", degenerate_candidate_kernel(outer, stride))
    assert_vector_axis(captured["VerifyParallelToPTO_before"], 0 if outer else 1)


@pytest.mark.pto
@pytest.mark.parametrize("name", DESIGN_CASES)
def test_design_case_uses_real_parallel_lowering(name):
    artifact, captured = lower_case(name, DESIGN_CASES[name])
    assert "VectorizeParallelToPTO_after" in captured
    after = captured["VectorizeParallelToPTO_after"]
    for func in after.functions.values():
        assert tvm.tirx.analysis.verify_ssa(func), "generated VMI binds must preserve SSA"
        definitions = list(func.params) + [buffer.data for buffer in func.buffer_map.values()]
        assert not tvm.tirx.analysis.undefined_vars(func.body, definitions), "conversion must consume source loop and proof variables"
    gather = calls(after, "tl.vmi.vgather")
    scatter = calls(after, "tl.vmi.vscatter")
    if name == "U2_dynamic_row":
        assert not gather and not scatter
    elif name in ("S1_scatter_f32", "S3_stride2"):
        assert scatter
    else:
        assert gather
    assert artifact.kernel_source
    assert not calls(after, "tl.vmi.vinterpret_cast"), "the PTOAS DSL normalizes same-width offset signedness"
    for call in gather:
        assert len(call.args) == 4
        assert str(call.args[2].dtype) in ("int16x128", "uint16x128", "int32x128", "uint32x128")
        assert str(call.args[3].dtype) == "boolx128"
    for call in scatter:
        assert len(call.args) == 5
        assert str(call.args[3].dtype) in ("int16x128", "uint16x128", "int32x128", "uint32x128")
        assert str(call.args[4].dtype) == "boolx128"


@pytest.mark.pto
@pytest.mark.parametrize("lanes,dtype", [(64, "float32"), (128, "float16"), (256, "bfloat16")])
@pytest.mark.parametrize("kind", ["gather", "scatter"])
def test_indexed_dtype_width_and_tail(lanes, dtype, kind):
    _, captured = lower_case(f"{kind}_{dtype}_{lanes}_tail", indexed_kernel(kind, dtype, lanes, lanes + 2))
    after = captured["VectorizeParallelToPTO_after"]
    operation = calls(after, "tl.vmi.vgather" if kind == "gather" else "tl.vmi.vscatter")
    assert operation
    index = 2 if kind == "gather" else 3
    assert str(operation[0].args[index].dtype) == f"{'int16' if dtype != 'float32' else 'int32'}x{lanes}"
    assert calls(after, "tl.vmi.vsel")


@pytest.mark.pto
def test_loaded_index_keeps_its_definition_value():
    _, captured = lower_case("saved_index_identity", saved_index)
    after = captured["VectorizeParallelToPTO_after"]
    index_loads = []
    for call in calls(after, "tl.vmi.vload"):
        ptr = call.args[0]
        if (
            getattr(ptr.op, "name", "") == "tl.access_ptr"
            and isinstance(ptr.args[0], tvm.tirx.BufferLoad)
            and ptr.args[0].buffer.name == "idx"
        ):
            index_loads.append(call)
    assert len(index_loads) == 1, "the old Bind must not reload idx after its store"


@pytest.mark.pto
@pytest.mark.parametrize("auto_schedule", [True, False])
def test_indexed_path_respects_schedule_switch(auto_schedule):
    func = indexed_kernel(manual_sync=not auto_schedule)
    artifact, captured = lower_case(f"gather_schedule_{auto_schedule}", func, auto_schedule=auto_schedule)
    assert calls(captured["VectorizeParallelToPTO_after"], "tl.vmi.vgather")
    assert "pto.set_flag(" in artifact.kernel_source and "pto.wait_flag(" in artifact.kernel_source


@pytest.mark.pto
def test_fragment_is_rejected():
    with pytest.raises(Exception, match=r"local.fragment|Fragment|only UB"):
        lower_case("rejected_fragment", rejected_fragment)


@pytest.mark.pto
def test_b16_wide_window_is_rejected_without_truncation():
    with pytest.raises(Exception, match=r"uint16|ui16|65536|window|offset"):
        lower_case("rejected_wide_window", rejected_wide_window)


@pytest.mark.pto
@pytest.mark.parametrize("access_kind", ["load", "store"])
def test_explicit_access_predicate_remains_outside_gather_scatter_scope(access_kind):
    func = indexed_kernel()

    def rewrite(node):
        if access_kind == "store" and isinstance(node, tvm.tirx.BufferStore) and node.buffer.name == "y":
            return tvm.tirx.BufferStore(node.buffer, node.value, node.indices, predicate=node.indices[0] < 64)
        if access_kind == "load" and isinstance(node, tvm.tirx.BufferLoad) and node.buffer.name == "idx":
            return tvm.tirx.BufferLoad(node.buffer, node.indices, predicate=node.indices[0] < 64)
        return None

    body = tvm.tirx.stmt_functor.ir_transform(func.body, None, rewrite, ["tirx.BufferLoad", "tirx.BufferStore"])
    diagnostic = (
        "buffer `x` load.*requires gather.*BufferStore access predicate" if access_kind == "store" else "non-trivial access predicate"
    )
    with pytest.raises(Exception, match=diagnostic):
        lower_case(f"rejected_{access_kind}_predicate", func.with_body(body))


def saved_offset_kernel():
    @T.prim_func
    def main(A: T.Tensor((256,), "float32"), I: T.Tensor((128,), "int32"), C: T.Tensor((128,), "float32")):
        with T.Kernel(1):
            table = T.alloc_shared((256,), "float32")
            idx = T.alloc_shared((128,), "int32")
            y = T.alloc_shared((128,), "float32")
            T.copy(A, table)
            T.copy(I, idx)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(128):
                    old = idx[i]
                    idx[i] = 0
                    y[i] = table[old + i]
            T.copy(y, C)

    return main


@pytest.mark.pto
def test_saved_varying_index_is_not_a_scalar_contiguous_base():
    _, captured = lower_case("saved_varying_plus_coordinate", saved_offset_kernel())
    after = captured["VectorizeParallelToPTO_after"]
    assert calls(after, "tl.vmi.vgather"), "the loaded Bind varies by lane even when added to a contiguous coordinate"


def b16_rebased_window_kernel():
    @T.prim_func
    def main(A: T.Tensor((70000,), "float16"), I: T.Tensor((128,), "uint16"), C: T.Tensor((128,), "float16")):
        with T.Kernel(1):
            raw = T.alloc_shared((70000,), "float16")
            view = tvm.tirx.decl_buffer((256,), "float16", data=raw.data, elem_offset=66000, scope="shared.dyn")
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
def test_b16_window_keeps_native_relative_index():
    _, captured = lower_case("b16_rebased_window", b16_rebased_window_kernel())
    gathered = calls(captured["VectorizeParallelToPTO_after"], "tl.vmi.vgather")
    assert len(gathered) == 1
    assert int(gathered[0].args[1]) == 66000
    assert str(gathered[0].args[2].dtype) == "uint16x128"
    assert not calls(captured["VectorizeParallelToPTO_after"], "tl.vmi.vcvt")


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
@pytest.mark.parametrize("extent", [22, 128, 150, 256, 278])
def test_tail_presence_selects_loop_wide_offset_cleaning(kind, extent):
    artifact, captured = lower_case(f"unified_{kind}_{extent}", indexed_kernel(kind, lanes=128, extent=extent))
    mod = captured["VectorizeParallelToPTO_after"]
    loops = []
    for func in mod.functions.values():
        tvm.tirx.stmt_functor.post_order_visit(func.body, lambda node: loops.append(node) if isinstance(node, tvm.tirx.For) else None)
    has_tail = extent % 128 != 0
    assert [int(loop.extent) for loop in loops] == [(extent + 127) // 128]
    masks = calls(mod, "tl.vmi.create_mask")
    assert len(masks) == 1
    if has_tail:
        assert isinstance(masks[0].args[0], tvm.tirx.BufferLoad), "the loop mask uses remaining for full and tail chunks"
        assert masks[0].args[0].buffer.name.startswith("remaining")
    else:
        assert int(masks[0].args[0]) == 128
    loop_operations = []
    tvm.tirx.stmt_functor.post_order_visit(
        loops[0].body, lambda node: loop_operations.append(node.op.name) if isinstance(node, tvm.tirx.Call) else None
    )
    assert loop_operations.count("tl.vmi.vsel") == int(has_tail)
    assert loop_operations.count("tl.vmi.vbrc") == int(has_tail), "only loops with a tail broadcast a cleaning zero"
    operation = calls(mod, "tl.vmi.vgather" if kind == "gather" else "tl.vmi.vscatter")
    assert len(operation) == 1, "one shared body handles every chunk"
    assert artifact.kernel_source


def mixed_indexed_units_kernel():
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
                for i in T.Parallel(150):
                    y[i] = x[idx[i]] + x[i]
                for j in T.Parallel(256):
                    y[j] = x[idx[j]] + y[j]
                for k in T.Parallel(150):
                    y[k] = y[k] + x[idx[k]]
            T.copy(y, C)

    return main


@pytest.mark.pto
def test_sequential_units_choose_offset_cleaning_independently():
    _, captured = lower_case("mixed_indexed_units", mixed_indexed_units_kernel())
    mod = captured["VectorizeParallelToPTO_after"]
    loops = []
    for func in mod.functions.values():
        tvm.tirx.stmt_functor.post_order_visit(func.body, lambda node: loops.append(node) if isinstance(node, tvm.tirx.For) else None)
    assert len(loops) == 3 and all(int(loop.extent) == 2 for loop in loops)
    for loop, has_tail in zip(loops, (True, False, True)):
        operations = []
        tvm.tirx.stmt_functor.post_order_visit(
            loop.body, lambda node, operations=operations: operations.append(node) if isinstance(node, tvm.tirx.Call) else None
        )
        names = [node.op.name for node in operations]
        assert names.count("tl.vmi.vsel") == int(has_tail)
        assert names.count("tl.vmi.vbrc") == int(has_tail)
        assert names.count("tl.vmi.create_mask") == int(has_tail)
        gather = next(node for node in operations if node.op.name == "tl.vmi.vgather")
        add = next(node for node in operations if node.op.name == "tl.vmi.vadd")
        store = next(node for node in operations if node.op.name == "tl.vmi.vstore")
        assert gather.args[3].same_as(add.args[2]) and gather.args[3].same_as(store.args[3])


def indexed_2d_tail_kernel(kind="gather"):
    @T.prim_func
    def main(A: T.Tensor((3, 256), "float32"), I: T.Tensor((3, 256), "int32"), C: T.Tensor((3, 256), "float32")):
        with T.Kernel(1):
            x = T.alloc_shared((3, 256), "float32")
            idx = T.alloc_shared((3, 256), "int32")
            y = T.alloc_shared((3, 256), "float32")
            T.copy(A, x)
            T.copy(I, idx)
            T.copy(C, y)
            with T.SimdVF(lanes=128):
                for i, j in T.Parallel(3, 150):
                    if kind == "gather":
                        y[i, j] = x[i, idx[i, j]]
                    else:
                        y[i, idx[i, j]] = x[i, j]
            T.copy(y, C)

    return main


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
def test_2d_indexed_tail_cleans_offsets_in_the_shared_loop(kind):
    _, captured = lower_case(f"unified_2d_{kind}", indexed_2d_tail_kernel(kind))
    mod = captured["VectorizeParallelToPTO_after"]
    assert len(calls(mod, "tl.vmi.vsel")) == 1
    masks = calls(mod, "tl.vmi.create_mask")
    assert len(masks) == 1 and isinstance(masks[0].args[0], tvm.tirx.BufferLoad)
    assert masks[0].args[0].buffer.name.startswith("remaining")
    operations = calls(mod, "tl.vmi.vgather" if kind == "gather" else "tl.vmi.vscatter")
    assert len(operations) == 1


def saved_index_loop_kernel():
    @T.prim_func
    def main(A: T.Tensor((384,), "float32"), I: T.Tensor((384,), "int32"), C: T.Tensor((384,), "float32")):
        with T.Kernel(1):
            table = T.alloc_shared((384,), "float32")
            idx = T.alloc_shared((384,), "int32")
            y = T.alloc_shared((384,), "float32")
            T.copy(A, table)
            T.copy(I, idx)
            T.copy(C, y)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(278):
                    old = idx[i]
                    idx[i] = 0
                    y[i] = table[old]
            T.copy(y, C)

    return main


@pytest.mark.pto
def test_unified_loop_keeps_the_saved_index_definition():
    _, captured = lower_case("unified_saved_index", saved_index_loop_kernel())
    mod = captured["VectorizeParallelToPTO_after"]
    index_loads = [call for call in calls(mod, "tl.vmi.vload") if call.args[0].args[0].buffer.name == "idx"]
    assert len(index_loads) == 1, "every loop iteration reads the old index only at its definition"
    assert len(calls(mod, "tl.vmi.vgather")) == 1
    cleaned = calls(mod, "tl.vmi.vsel")
    assert len(cleaned) == 1
    assert isinstance(cleaned[0].args[1], tvm.tirx.Var) and cleaned[0].args[1].name == "old"


def handwritten_signed_offset_kernel(kind="gather", index_dtype="int32", payload_dtype="float32", owner_size=256, *, compact=False):
    source_size = 128 if compact and kind == "scatter" else owner_size
    output_size = 128 if compact and kind == "gather" else owner_size

    @T.prim_func
    def main(A: T.Tensor((source_size,), payload_dtype), I: T.Tensor((128,), index_dtype), C: T.Tensor((output_size,), payload_dtype)):
        with T.Kernel(1):
            x = T.alloc_shared((source_size,), payload_dtype)
            idx = T.alloc_shared((128,), index_dtype)
            y = T.alloc_shared((output_size,), payload_dtype)
            T.copy(A, x)
            T.copy(I, idx)
            T.copy(C, y)
            with T.SimdVF(lanes=128):
                mask = T.vmi.create_mask(128, size=128)
                offsets = T.vmi.vload(idx[0], size=128)
                if kind == "gather":
                    value = T.vmi.vgather(x[0], offsets, mask, pmode="zero")
                    T.vmi.vstore(value, y[0], mask)
                else:
                    value = T.vmi.vload(x[0], size=128)
                    T.vmi.vscatter(value, y[0], offsets, mask)
            T.copy(y, C)

    return main


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
@pytest.mark.parametrize(
    "index_dtype,payload_dtype", [("int32", "float32"), ("int16", "float16"), ("int8", "float32"), ("int8", "float16")]
)
def test_handwritten_signed_offsets_reach_ptoas_without_tilelang_reinterpret(kind, index_dtype, payload_dtype):
    artifact = lower(handwritten_signed_offset_kernel(kind, index_dtype, payload_dtype), target="pto")
    assert "pto.vmi.vinterpret_cast(" not in artifact.kernel_source
    assert "pto.vmi.vgather(" in artifact.kernel_source or "pto.vmi.vscatter(" in artifact.kernel_source


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
@pytest.mark.parametrize("index_dtype", ["int16", "uint16"])
@pytest.mark.parametrize("extent", [256, 278])
def test_native_b16_index_does_not_round_trip_through_int32(kind, index_dtype, extent):
    artifact, captured = lower_case(
        f"native16_{kind}_{index_dtype}_{extent}",
        indexed_kernel(kind, "float16", 128, extent, index_dtype=index_dtype),
    )
    mod = captured["VectorizeParallelToPTO_after"]
    assert not calls(mod, "tl.vmi.vcvt"), "native B16 offsets must not widen then narrow"
    assert not calls(mod, "tl.vmi.vinterpret_cast"), "signedness belongs to the PTOAS DSL"
    op = "tl.vmi.vgather" if kind == "gather" else "tl.vmi.vscatter"
    offset_arg = 2 if kind == "gather" else 3
    operations = calls(mod, op)
    assert operations
    assert all(str(call.args[offset_arg].dtype) == f"{index_dtype}x128" for call in operations)
    assert "pto.vmi.vcvt(" not in artifact.kernel_source


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
@pytest.mark.parametrize("index_dtype", ["int16", "uint16"])
@pytest.mark.parametrize("extent", [256, 150])
def test_b32_memory_delegates_native_int16_index_widening_to_dsl(kind, index_dtype, extent):
    artifact, captured = lower_case(
        f"b32_native16_{kind}_{index_dtype}_{extent}", indexed_kernel(kind, "float32", 128, extent, index_dtype=index_dtype)
    )
    mod = captured["VectorizeParallelToPTO_after"]
    assert not calls(mod, "tl.vmi.vcvt"), "PTOAS normalizes native 16-bit offsets before widening to ui32"
    assert not calls(mod, "tl.vmi.vinterpret_cast")
    operations = calls(mod, "tl.vmi.vgather" if kind == "gather" else "tl.vmi.vscatter")
    offset_arg = 2 if kind == "gather" else 3
    assert all(str(call.args[offset_arg].dtype) == f"{index_dtype}x128" for call in operations)
    assert "pto.vmi.vcvt(" not in artifact.kernel_source


def explicit_wide_index_kernel(kind="gather", payload_dtype="float32"):
    @T.prim_func
    def main(A: T.Tensor((256,), payload_dtype), I: T.Tensor((256,), "int16"), C: T.Tensor((256,), payload_dtype)):
        with T.Kernel(1):
            x = T.alloc_shared((256,), payload_dtype)
            idx = T.alloc_shared((256,), "int16")
            y = T.alloc_shared((256,), payload_dtype)
            T.copy(A, x)
            T.copy(I, idx)
            T.copy(C, y)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(150):
                    if kind == "gather":
                        y[i] = x[T.Cast("int32", idx[i])]
                    else:
                        y[T.Cast("int32", idx[i])] = x[i]
            T.copy(y, C)

    return main


def explicit_narrow_index_kernel(kind="gather", extent=150):
    @T.prim_func
    def main(A: T.Tensor((256,), "float16"), I: T.Tensor((256,), "int32"), C: T.Tensor((256,), "float16")):
        with T.Kernel(1):
            x = T.alloc_shared((256,), "float16")
            idx = T.alloc_shared((256,), "int32")
            y = T.alloc_shared((256,), "float16")
            T.copy(A, x)
            T.copy(I, idx)
            T.copy(C, y)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(extent):
                    if kind == "gather":
                        y[i] = x[T.Cast("uint16", idx[i])]
                    else:
                        y[T.Cast("uint16", idx[i])] = x[i]
            T.copy(y, C)

    return main


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
@pytest.mark.parametrize("index_dtype", ["int32", "uint32"])
@pytest.mark.parametrize("extent", [256, 150])
def test_b16_access_rejects_implicit_offset_narrowing(kind, index_dtype, extent):
    with pytest.raises(Exception, match=r"32-bit offsets require implicit narrowing to 16 bits"):
        lower_case(
            f"rejected_narrow_{kind}_{index_dtype}_{extent}",
            indexed_kernel(kind, "float16", 128, extent, index_dtype=index_dtype),
        )


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
def test_b16_access_preserves_explicit_source_narrowing(kind):
    _, captured = lower_case(f"explicit_narrow_{kind}", explicit_narrow_index_kernel(kind))
    mod = captured["VectorizeParallelToPTO_after"]
    conversions = calls(mod, "tl.vmi.vcvt")
    assert len(conversions) == 1 and str(conversions[0].dtype) == "uint16x128"


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
def test_b16_access_does_not_erase_explicit_source_widening(kind):
    with pytest.raises(Exception, match="implicit narrowing"):
        lower_case(f"rejected_explicit_wide_{kind}", explicit_wide_index_kernel(kind, "float16"))


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
def test_explicit_source_index_widening_is_preserved(kind):
    _, captured = lower_case(f"b32_explicit_widen_{kind}", explicit_wide_index_kernel(kind))
    mod = captured["VectorizeParallelToPTO_after"]
    conversions = calls(mod, "tl.vmi.vcvt")
    assert len(conversions) == 1 and str(conversions[0].dtype) == "int32x128"


def native16_offset_view_kernel(kind="gather", payload_dtype="float16"):
    source_size = 40400 if kind == "gather" else 384
    output_size = 384 if kind == "gather" else 40400

    @T.prim_func
    def main(A: T.Tensor((source_size,), payload_dtype), I: T.Tensor((384,), "uint16"), C: T.Tensor((output_size,), payload_dtype)):
        with T.Kernel(1):
            x = T.alloc_shared((source_size,), payload_dtype)
            idx = T.alloc_shared((384,), "uint16")
            y = T.alloc_shared((output_size,), payload_dtype)
            x_view = tvm.tirx.decl_buffer((384,), payload_dtype, data=x.data, elem_offset=40000, scope="shared.dyn")
            y_view = tvm.tirx.decl_buffer((384,), payload_dtype, data=y.data, elem_offset=40000, scope="shared.dyn")
            T.copy(A, x)
            T.copy(I, idx)
            T.copy(C, y)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(278):
                    if kind == "gather":
                        y[i] = x_view[idx[i]]
                    else:
                        y_view[idx[i]] = x[i]
            T.copy(y, C)

    return main


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
@pytest.mark.parametrize("payload_dtype", ["float16", "float32"])
def test_native16_address_arithmetic_keeps_wide_intermediates(kind, payload_dtype):
    if payload_dtype == "float16":
        with pytest.raises(Exception, match="implicit narrowing"):
            lower_case(f"native16_{kind}_{payload_dtype}_offset_view", native16_offset_view_kernel(kind, payload_dtype))
        return
    _, captured = lower_case(f"native16_{kind}_{payload_dtype}_offset_view", native16_offset_view_kernel(kind, payload_dtype))
    mod = captured["VectorizeParallelToPTO_after"]
    additions = calls(mod, "tl.vmi.vadd")
    assert additions and all(str(call.dtype) == "int32x128" for call in additions)
    assert calls(mod, "tl.vmi.vcvt"), "adding the view origin must not wrap in 16-bit arithmetic"
