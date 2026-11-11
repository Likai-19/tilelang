"""Lexical allocation-owner visibility for automatic indexed Parallel access."""

import importlib.util
from pathlib import Path

import pytest

from tilelang import tvm
import tilelang.ascend.language as T

_spec = importlib.util.spec_from_file_location(
    "pto_indexed_scope_contract", Path(__file__).with_name("test_pto_parallel_gather_scatter.py")
)
gs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gs)


@T.prim_func
def nested_scatter_owner(A: T.Tensor((256,), "float32"), I: T.Tensor((256,), "int32")):
    with T.Kernel(1):
        x = T.alloc_shared((256,), "float32")
        idx = T.alloc_shared((256,), "int32")
        T.copy(A, x)
        T.copy(I, idx)
        with T.SimdVF(lanes=128):  # noqa: SIM117 - nested block is the scope under test
            with T.sblock("indexed_owner", no_realize=True):
                y = T.alloc_shared((256,), "float32")
                for i in T.Parallel(256):
                    y[idx[i]] = x[i]


def _allocation_form(form):
    def rewrite(node):
        if not isinstance(node, tvm.tirx.SBlock) or node.name_hint != "indexed_owner":
            return None
        if form == "block":
            return node
        if form == "realize":
            return tvm.tirx.SBlockRealize([], True, node)
        assert form == "flat"
        body = tvm.tirx.SeqStmt([tvm.tirx.AllocBuffer(buffer) for buffer in node.alloc_buffers] + [node.body])
        return tvm.tirx.SBlock(node.iter_vars, node.reads, node.writes, node.name_hint, body, annotations=node.annotations)

    body = tvm.tirx.stmt_functor.ir_transform(nested_scatter_owner.body, None, rewrite, ["tirx.SBlock"])
    return nested_scatter_owner.with_body(body)


@pytest.mark.pto
@pytest.mark.parametrize("form", ["block", "realize", "flat"])
def test_nested_allocation_is_visible_to_indexed_parallel(form):
    artifact, captured = gs.lower_case(f"nested_indexed_owner_{form}", _allocation_form(form))
    scatters = gs.calls(captured["VectorizeParallelToPTO_after"], "tl.vmi.vscatter")
    assert len(scatters) == 1
    pointer = scatters[0].args[1]
    assert pointer.op.name == "tl.access_ptr"
    assert pointer.args[0].buffer.name == "y"
    assert int(pointer.args[1]) == 256
    assert int(pointer.args[2]) == 2
    assert "pto.vmi.vscatter(" in artifact.kernel_source


@T.prim_func
def nested_scatter_then_read(A: T.Tensor((256,), "float32"), I: T.Tensor((256,), "int32"), C: T.Tensor((256,), "float32")):
    with T.Kernel(1):
        x = T.alloc_shared((256,), "float32")
        idx = T.alloc_shared((256,), "int32")
        y = T.alloc_shared((256,), "float32")
        T.copy(A, x)
        T.copy(I, idx)
        with T.SimdVF(lanes=128):  # noqa: SIM117 - nested owner is the scope under test
            with T.sblock("indexed_owner", no_realize=True):
                scratch = T.alloc_shared((256,), "float32")
                for i in T.Parallel(256):
                    scratch[idx[i]] = x[i]
                for j in T.Parallel(256):
                    y[j] = scratch[j]
        T.copy(y, C)
