"""Automatic indexed memory admission, owner descriptors, and scheduling."""

import importlib.util
from pathlib import Path

import pytest

import tilelang.ascend.language as T
from tilelang import tvm
from testing.ascend.auto_schedule._reuse_utils import _insert_sync_alias_pairs

_spec = importlib.util.spec_from_file_location("indexed_contract_gs", Path(__file__).with_name("test_pto_parallel_gather_scatter.py"))
gs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gs)


@pytest.mark.pto
@pytest.mark.parametrize(
    "payload,index,diagnostic",
    [("int64", "int32", "element dtype int64"), ("float32", "int64", "element dtype int64"), ("float32", "int8", "16-bit or 32-bit")],
)
def test_automatic_indexed_type_limits(payload, index, diagnostic):
    with pytest.raises(Exception, match=diagnostic):
        gs.lower_case(f"rejected_{payload}_{index}", gs.indexed_kernel(dtype=payload, index_dtype=index))


def _non_ub_index_kernel(index_scope):
    @T.prim_func
    def main(A: T.Tensor((256,), "float32"), I: T.Tensor((256,), "int32"), C: T.Tensor((256,), "float32")):
        with T.Kernel(1):
            x = T.alloc_shared((256,), "float32")
            idx = T.alloc_fragment((256,), "int32")
            y = T.alloc_shared((256,), "float32")
            T.copy(A, x)
            if index_scope == "fragment":
                T.copy(I, idx)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(256):
                    if index_scope == "fragment":
                        y[i] = x[idx[i]]
                    else:
                        y[i] = x[I[i]]
            T.copy(y, C)

    return main


@pytest.mark.pto
@pytest.mark.parametrize("scope,diagnostic", [("fragment", "local.fragment|only UB"), ("global", "global.*SIMD_VF|SIMD_VF.*global")])
def test_index_must_be_staged_in_ordinary_ub(scope, diagnostic):
    with pytest.raises(Exception, match=diagnostic):
        gs.lower_case(f"rejected_{scope}_index", _non_ub_index_kernel(scope))


@pytest.mark.pto
@pytest.mark.parametrize("kind,diagnostic", [("volatile", "volatile"), ("opaque", "unsupported|opaque")])
def test_indexed_access_does_not_drop_memory_effects(kind, diagnostic):
    func = gs.indexed_kernel()

    def rewrite(node):
        if kind == "volatile" and isinstance(node, tvm.tirx.SBlock) and node.name_hint == "SIMD_VF":
            return tvm.tirx.SBlock(
                node.iter_vars,
                node.reads,
                node.writes,
                node.name_hint,
                tvm.tirx.AttrStmt(tvm.tirx.IntImm("int32", 0), "volatile_scope", 1, node.body),
                node.init,
                node.alloc_buffers,
                node.match_buffers,
                node.annotations,
            )
        if (
            kind == "opaque"
            and isinstance(node, tvm.tirx.BufferLoad)
            and node.buffer.name == "x"
            and isinstance(node.indices[0], tvm.tirx.BufferLoad)
        ):
            return tvm.tirx.BufferLoad(node.buffer, [tvm.tirx.call_extern("int32", "read_index", node.indices[0])])
        return None

    body = tvm.tirx.stmt_functor.ir_transform(func.body, None, rewrite, ["tirx.SBlock", "tirx.BufferLoad"])
    with pytest.raises(Exception, match=diagnostic):
        gs.lower_case(f"rejected_{kind}_index", func.with_body(body))


def _view_kernel(kind):
    @T.prim_func
    def main(A: T.Tensor((800,), "float32"), I: T.Tensor((256,), "int32"), C: T.Tensor((800,), "float32")):
        with T.Kernel(1):
            x = T.alloc_shared((800,), "float32")
            y = T.alloc_shared((800,), "float32")
            idx = T.alloc_shared((256,), "int32")
            x_view = tvm.tirx.decl_buffer((2, 256), "float32", data=x.data, strides=(512, 1), elem_offset=32, scope="shared.dyn")
            y_view = tvm.tirx.decl_buffer((2, 256), "float32", data=y.data, strides=(512, 1), elem_offset=32, scope="shared.dyn")
            T.copy(A, x)
            T.copy(I, idx)
            T.copy(C, y)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(256):
                    if kind == "gather":
                        y[i] = x_view[1, idx[i]]
                    else:
                        y_view[1, idx[i]] = x[i]
            T.copy(y, C)

    return main


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
def test_late_indexed_descriptor_uses_whole_owner_once(kind):
    _, captured = gs.lower_case(f"owner_view_{kind}", _view_kernel(kind))
    mod = captured["VectorizeParallelToPTO_after"]
    op = "tl.vmi.vgather" if kind == "gather" else "tl.vmi.vscatter"
    call = gs.calls(mod, op)[0]
    pointer = call.args[0 if kind == "gather" else 1]
    assert pointer.op.name == "tl.access_ptr"
    origin, extent, mode = pointer.args
    assert origin.buffer.name == ("x" if kind == "gather" else "y")
    assert [int(dim) for dim in origin.buffer.shape] == [800]
    assert [int(index) for index in origin.indices] == [0]
    assert int(extent) == 800 and int(mode) == (1 if kind == "gather" else 2)
    assert int(call.args[1 if kind == "gather" else 2]) == 0
    broadcasts = gs.calls(mod, "tl.vmi.vbrc")
    assert [int(node.args[0]) for node in broadcasts] == [544], "view origin 32 plus row stride 512 enters the vector offset once"
    assert len(gs.calls(mod, "tl.vmi.vadd")) == 1


def _scatter_reuse_program():
    @T.prim_func
    def main(
        initial: T.Tensor((256,), "float32"),
        values: T.Tensor((4, 256), "float32"),
        indices: T.Tensor((256,), "int32"),
        out_a: T.Tensor((4,), "float32"),
        out_b: T.Tensor((4, 256), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((256,), "float32")
            b = T.alloc_shared((256,), "float32")
            idx = T.alloc_shared((256,), "int32")
            T.copy(initial, a)
            T.copy(indices, idx)
            for k in T.Pipelined(4, num_stages=1):
                out_a[k] = a[255]
                T.copy(values[k, :], b)
                T.copy(b, out_b[k, :])
                with T.SimdVF(lanes=128):
                    for i in T.Parallel(128):
                        a[idx[i]] = b[i]

    return main


@pytest.mark.pto
def test_scatter_may_write_does_not_kill_live_owner_elements():
    # A legal input can leave a[255] untouched in every iteration. Its old
    # value stays live across b's writes and must prevent allocation reuse.
    assert ("a", "b") not in _insert_sync_alias_pairs(_scatter_reuse_program())


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
def test_b16_window_span_cannot_wrap_into_uint16(kind):
    source_size = 70000 if kind == "gather" else 128
    output_size = 128 if kind == "gather" else 70000

    @T.prim_func
    def main(A: T.Tensor((source_size,), "float16"), I: T.Tensor((128,), "int32"), C: T.Tensor((output_size,), "float16")):
        with T.Kernel(1):
            x = T.alloc_shared((source_size,), "float16")
            y = T.alloc_shared((output_size,), "float16")
            idx = T.alloc_shared((128,), "int32")
            x_view = tvm.tirx.decl_buffer((65537,), "float16", data=x.data, strides=(65536,), scope="shared.dyn")
            y_view = tvm.tirx.decl_buffer((65537,), "float16", data=y.data, strides=(65536,), scope="shared.dyn")
            T.copy(A, x)
            T.copy(I, idx)
            T.copy(C, y)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(128):
                    if kind == "gather":
                        y[i] = x_view[idx[i]]
                    else:
                        y_view[idx[i]] = x[i]
            T.copy(y, C)

    # idx=1 is a legal active access to owner element 65536. The physical
    # window is 2**32 + 1, which used to wrap to 1 and truncate this offset.
    with pytest.raises(Exception, match="ui16 relative window"):
        gs.lower_case(f"overflow_window_{kind}", main)
