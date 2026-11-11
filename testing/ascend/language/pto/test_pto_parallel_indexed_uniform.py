"""Uniform scatter targets preserve scalar addressing and SSA."""

import importlib.util
from pathlib import Path

import pytest

import tilelang.ascend.language as T
from tilelang import tvm

_spec = importlib.util.spec_from_file_location("indexed_uniform_gs", Path(__file__).with_name("test_pto_parallel_gather_scatter.py"))
gs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gs)


def uniform_store_kernel(value_kind="vector", extent=256, *, select_target=False):
    @T.prim_func
    def main(A: T.Tensor((256,), "int32"), C: T.Tensor((512,), "int32")):
        with T.Kernel(1):
            x = T.alloc_shared((256,), "int32")
            y = T.alloc_shared((512,), "int32")
            T.copy(A, x)
            T.copy(C, y)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(extent):
                    if select_target:
                        y[T.Select(x[0] > 0, 131, 135)] = x[i]
                    elif value_kind == "scalar":
                        y[131] = 7
                    else:
                        y[131] = x[i]
            T.copy(y, C)

    return main


@pytest.mark.pto
@pytest.mark.parametrize("value_kind", ["scalar", "vector"])
@pytest.mark.parametrize("extent", [256, 150])
def test_uniform_store_broadcasts_actual_nonzero_offset(value_kind, extent):
    _, captured = gs.lower_case(f"uniform_store_{value_kind}_{extent}", uniform_store_kernel(value_kind, extent))
    mod = captured["VectorizeParallelToPTO_after"]
    scatter = gs.calls(mod, "tl.vmi.vscatter")
    assert len(scatter) == 1, "duplicate active targets are admitted as scatter"
    scatter = scatter[0]
    pointer = scatter.args[1]
    assert pointer.args[0].buffer.name == "y"
    assert int(pointer.args[0].indices[0]) == 0
    assert int(pointer.args[1]) == 512, "the descriptor covers the actual owner"
    assert int(pointer.args[2]) == 2
    assert int(scatter.args[2]) == 0
    assert str(scatter.args[0].dtype) == str(scatter.args[3].dtype) == "int32x128"
    assert str(scatter.args[4].dtype) == "boolx128"
    bindings = {}
    for func in mod.functions.values():
        tvm.tirx.stmt_functor.post_order_visit(
            func.body, lambda node: bindings.update({node.var: node.value}) if isinstance(node, tvm.tirx.Bind) else None
        )
    offsets = bindings[scatter.args[3]]
    if extent == 150:
        assert offsets.op.name == "tl.vmi.vsel"
        assert offsets.args[0].same_as(scatter.args[4])
        zero = bindings[offsets.args[2]]
        assert zero.op.name == "tl.vmi.vbrc" and int(zero.args[0]) == 0
        offsets = bindings[offsets.args[1]]
    assert offsets.op.name == "tl.vmi.vbrc"
    assert int(offsets.args[0]) == 131, "scatter consumes the real nonzero uniform offset"
    if value_kind == "scalar":
        value = bindings[scatter.args[0]]
        assert value.op.name == "tl.vmi.vbrc" and int(value.args[0]) == 7
    loads = gs.calls(mod, "tl.vmi.vload")
    assert len(loads) == int(value_kind == "vector")
    assert len(gs.calls(mod, "tl.vmi.vsel")) == int(extent == 150), "offset cleaning follows the unit's tail policy"
    assert not gs.calls(mod, "tl.vmi.vgather")
    assert not gs.calls(mod, "tl.vmi.vstore")


@pytest.mark.pto
def test_unconditional_uniform_store_keeps_scalar_select_target():
    _, captured = gs.lower_case("uniform_store_scalar_select_target", uniform_store_kernel(select_target=True))
    mod = captured["VectorizeParallelToPTO_after"]
    scatter = gs.calls(mod, "tl.vmi.vscatter")
    assert len(scatter) == 1
    targets = [call.args[0] for call in gs.calls(mod, "tl.vmi.vbrc") if isinstance(call.args[0], tvm.tirx.Select)]
    assert len(targets) == 1
    assert int(targets[0].true_value) == 131 and int(targets[0].false_value) == 135
    reads = []
    tvm.tirx.stmt_functor.post_order_visit(
        targets[0].condition, lambda node: reads.append(node) if isinstance(node, tvm.tirx.BufferLoad) else None
    )
    assert len(reads) == 1 and reads[0].buffer.name == "x"
    assert int(reads[0].indices[0]) == 0
    assert not gs.calls(mod, "tl.vmi.vgather")
    assert not gs.calls(mod, "tl.vmi.vsel"), "the scalar address Select does not create a conditional scatter mask"
