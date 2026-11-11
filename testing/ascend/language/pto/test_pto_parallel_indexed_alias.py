"""Saved indexed values retain SSA semantics during scalarization and cleanup."""

import pytest

from test_pto_parallel_gather_scatter import ascend_transform, calls, indexed_kernel, lower_case, tvm
from test_pto_parallel_indexed_control_flow import replace_unit_body
from tilelang.ascend.target import normalize_pto_target


@pytest.fixture(scope="module")
def source_module():
    _, captured = lower_case("indexed_alias_base", indexed_kernel(extent=150))
    return captured["VerifyParallelToPTO_before"]


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["uniform", "continuous", "alias", "scalar_read"])
def test_saved_index_cancellation_scalarizes_the_address(source_module, kind):
    def make_body(i, payload, index, output):
        saved = tvm.tirx.Var("saved", "int32")
        offset = tvm.tirx.Var("offset", "int32")
        statements = [tvm.tirx.Bind(saved, index[i]), tvm.tirx.BufferStore(index, 0, [i])]
        expression = tvm.tirx.Sub(saved, saved)
        if kind == "scalar_read":
            expression = tvm.tirx.Add(expression, index[0])
        statements.append(tvm.tirx.Bind(offset, expression))
        if kind == "alias":
            alias = tvm.tirx.Var("offset_alias", "int32")
            statements.append(tvm.tirx.Bind(alias, tvm.tirx.Add(offset, 0)))
            offset = alias
        address = offset if kind in ("uniform", "scalar_read") else tvm.tirx.Add(offset, i)
        statements.append(tvm.tirx.BufferStore(output, payload[address], [i]))
        return tvm.tirx.SeqStmt(statements)

    mod = replace_unit_body(source_module, make_body)
    with normalize_pto_target("pto"):
        verified = ascend_transform.VerifyParallelToPTO()(mod)
        converted = ascend_transform.VectorizeParallelToPTO()(verified)
    assert not calls(converted, "tl.vmi.vgather")
    loads = calls(converted, "tl.vmi.vload")
    if kind in ("uniform", "scalar_read"):
        assert not loads
        scalar_loads = []
        for func in converted.functions.values():
            tvm.tirx.stmt_functor.post_order_visit(
                func.body,
                lambda node: (
                    scalar_loads.append(node.value)
                    if isinstance(node, tvm.tirx.Bind) and isinstance(node.value, tvm.tirx.BufferLoad) and node.value.buffer.name == "x"
                    else None
                ),
            )
        assert len(scalar_loads) == 1
        if kind == "uniform":
            definitions = {}
            for func in converted.functions.values():
                tvm.tirx.stmt_functor.post_order_visit(
                    func.body, lambda node: definitions.update({node.var: node.value}) if isinstance(node, tvm.tirx.Bind) else None
                )
            scalar_offset = definitions[scalar_loads[0].indices[0]]
            assert int(scalar_offset) == 0
    else:
        assert len(loads) == 1
        pointer = loads[0].args[0]
        start = pointer.args[0].indices[0]
        assert "q" in str(start) and "128" in str(start)
    assert not calls(converted, "tl.vmi.vsub"), "the cancelled saved value must not retain its original read"


@pytest.mark.pto
def test_dead_gather_removes_its_index_and_tail_cleanup(source_module):
    def make_body(i, payload, index, output):
        dead = tvm.tirx.Var("dead", "float32")
        return tvm.tirx.SeqStmt([tvm.tirx.Bind(dead, payload[index[i]]), tvm.tirx.BufferStore(output, payload[i], [i])])

    mod = replace_unit_body(source_module, make_body)
    with normalize_pto_target("pto"):
        converted = ascend_transform.VectorizeParallelToPTO()(ascend_transform.VerifyParallelToPTO()(mod))
    assert len(calls(converted, "tl.vmi.vload")) == 1
    for operation in ("vgather", "vsel"):
        assert not calls(converted, "tl.vmi." + operation), "an unused gather must not keep its offset chain alive"
