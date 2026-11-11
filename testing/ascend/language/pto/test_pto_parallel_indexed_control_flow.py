"""Reject conditional indexed accesses in source TIR before legalization.

Construct units at the Verify boundary so earlier simplification cannot erase
Select/If/predicate operands. PR2 lowering is not present on this branch;
continuous control-flow neighbors retain their existing unsupported diagnostic.
"""

import pytest

from test_pto_parallel_gather_scatter import ascend_transform, indexed_kernel, lower_case, tvm
from tilelang.ascend.target import normalize_pto_target


@pytest.fixture(scope="module")
def source_module():
    _, captured = lower_case("conditional_indexed_base", indexed_kernel())
    return captured["VerifyParallelToPTO_before"]


def replace_unit_body(mod, make_body):
    def rewrite(node):
        if not isinstance(node, tvm.tirx.For) or "pto_parallel_loop_layout" not in node.annotations:
            return None
        store = node.body
        assert isinstance(store, tvm.tirx.BufferStore)
        payload = store.value.buffer
        index = store.value.indices[0].buffer
        return tvm.tirx.For(
            node.loop_var,
            node.min,
            node.extent,
            node.kind,
            make_body(node.loop_var, payload, index, store.buffer),
            annotations=node.annotations,
        )

    return tvm.IRModule(
        {
            name: func.with_body(tvm.tirx.stmt_functor.ir_transform(func.body, None, rewrite, ["tirx.For"]))
            for name, func in mod.functions.items()
        }
    )


def verify_body(mod, make_body):
    mod = replace_unit_body(mod, make_body)
    with normalize_pto_target("pto"):
        result = ascend_transform.VerifyParallelToPTO()(mod)
    assert tvm.ir.structural_equal(mod, result), "Verify must preserve source evaluation and access positions"


def branch_body(kind, location):
    def make_body(i, payload, index, output):
        condition = i < 150
        access = tvm.tirx.BufferLoad(payload, [index[i]])
        if kind == "nested_index":
            access = tvm.tirx.BufferLoad(payload, [index[index[i]]])
        if kind in ("gather", "nested_index"):
            statement = tvm.tirx.BufferStore(output, access, [i])
        elif kind == "bind":
            saved = tvm.tirx.Var("saved", "float32")
            statement = tvm.tirx.SeqStmt([tvm.tirx.Bind(saved, access), tvm.tirx.BufferStore(output, saved, [i])])
        elif kind == "scatter":
            statement = tvm.tirx.BufferStore(output, payload[i], [index[i]])
        else:
            statement = tvm.tirx.BufferStore(output, payload[i], [0])
        if location == "condition":
            return tvm.tirx.IfThenElse(access > 0, tvm.tirx.BufferStore(output, payload[i], [i]), None)
        if location == "then":
            return tvm.tirx.IfThenElse(condition, statement, None)
        return tvm.tirx.IfThenElse(condition, tvm.tirx.BufferStore(output, payload[i], [i]), statement)

    return make_body


@pytest.mark.pto
@pytest.mark.parametrize(
    "kind,location,buffer,operation",
    [
        ("gather", "condition", "x", "gather"),
        ("gather", "then", "x", "gather"),
        ("gather", "else", "x", "gather"),
        ("bind", "then", "x", "gather"),
        ("nested_index", "else", "idx", "gather"),
        ("scatter", "then", "y", "scatter"),
        ("scatter", "else", "y", "scatter"),
        ("uniform_store", "then", "y", "scatter"),
    ],
)
def test_branch_reports_original_indexed_access(source_module, kind, location, buffer, operation):
    with pytest.raises(Exception) as error:
        verify_body(source_module, branch_body(kind, location))
    message = str(error.value)
    assert "[VerifyParallelToPTO] control flow:" in message
    assert f"buffer `{buffer}` {'store' if operation == 'scatter' else 'load'}" in message
    assert f"requires {operation}" in message
    assert f"IfThenElse {location}" in message
    assert "region #0, unit #0" in message
    if kind == "bind":
        assert "Bind `saved` value" in message


@pytest.mark.pto
@pytest.mark.parametrize("location", ["condition", "true value", "false value", "nested index", "unconditional_scatter_index"])
def test_select_does_not_protect_indexed_reads(source_module, location):
    def make_body(i, payload, index, output):
        gather = payload[index[i]]
        condition = gather > 0 if location == "condition" else i < 150
        true_value = gather if location == "true value" else payload[i]
        false_value = gather if location == "false value" else tvm.tirx.FloatImm("float32", 0)
        if location == "nested index":
            true_value = payload[index[index[i]]]
        if location == "unconditional_scatter_index":
            offset = tvm.tirx.Select(i < 150, index[index[i]], index[i])
            return tvm.tirx.BufferStore(output, payload[i], [offset])
        return tvm.tirx.BufferStore(output, tvm.tirx.Select(condition, true_value, false_value), [i])

    with pytest.raises(Exception) as error:
        verify_body(source_module, make_body)
    message = str(error.value)
    assert "[VerifyParallelToPTO] control flow:" in message
    assert "requires gather" in message
    nested = location in ("nested index", "unconditional_scatter_index")
    assert f"Select {'true value' if nested else location}" in message
    assert f"buffer `{'idx' if nested else 'x'}` load" in message


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["load", "store", "uniform_store", "predicate_read", "nested_index"])
def test_access_predicate_reports_conditional_indexed_access(source_module, kind):
    def make_body(i, payload, index, output):
        if kind == "load":
            value = tvm.tirx.BufferLoad(payload, [index[i]], predicate=i < 150)
            return tvm.tirx.BufferStore(output, value, [i])
        if kind == "store":
            return tvm.tirx.BufferStore(output, payload[i], [index[i]], predicate=i < 150)
        if kind == "uniform_store":
            return tvm.tirx.BufferStore(output, payload[i], [0], predicate=i < 150)
        if kind == "nested_index":
            return tvm.tirx.BufferStore(output, payload[index[index[i]]], [i], predicate=i < 150)
        return tvm.tirx.BufferStore(output, payload[i], [i], predicate=payload[index[i]] > 0)

    with pytest.raises(Exception) as error:
        verify_body(source_module, make_body)
    message = str(error.value)
    assert "[VerifyParallelToPTO] control flow:" in message
    assert "access predicate" in message
    assert f"requires {'scatter' if kind in ('store', 'uniform_store') else 'gather'}" in message


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["branch", "select", "scatter_value_select", "predicate"])
def test_unconditional_gather_keeps_its_execution_position(source_module, kind):
    def make_body(i, payload, index, output):
        saved = tvm.tirx.Var("saved", "float32")
        bind = tvm.tirx.Bind(saved, payload[index[i]])
        store = tvm.tirx.BufferStore(output, saved, [i])
        if kind == "branch":
            store = tvm.tirx.IfThenElse(i < 150, store, None)
        elif kind == "select":
            store = tvm.tirx.BufferStore(output, tvm.tirx.Select(i < 150, saved, tvm.tirx.FloatImm("float32", 0)), [i])
        elif kind == "scatter_value_select":
            store = tvm.tirx.BufferStore(output, tvm.tirx.Select(i < 150, saved, tvm.tirx.FloatImm("float32", 0)), [index[i]])
        else:
            store = tvm.tirx.BufferStore(output, saved, [i], predicate=i < 150)
        return tvm.tirx.SeqStmt([bind, store])

    with pytest.raises(Exception) as error:
        verify_body(source_module, make_body)
    message = str(error.value)
    assert "[VerifyParallelToPTO] control flow:" not in message
    assert {"branch": "conditional branches", "select": "Select", "scatter_value_select": "Select", "predicate": "access predicate"}[
        kind
    ] in message


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["branch", "select", "predicate", "unsupported_call"])
def test_continuous_control_flow_keeps_existing_contract(source_module, kind):
    def make_body(i, payload, index, output):
        store = tvm.tirx.BufferStore(output, payload[i], [i])
        if kind == "branch":
            return tvm.tirx.IfThenElse(i < 150, store, None)
        if kind == "predicate":
            return tvm.tirx.BufferStore(output, payload[i], [i], predicate=i < 150)
        if kind == "unsupported_call":
            value = tvm.tirx.if_then_else(i < 150, payload[index[i]], tvm.tirx.FloatImm("float32", 0))
        else:
            value = tvm.tirx.Select(i < 150, payload[i], tvm.tirx.FloatImm("float32", 0))
        return tvm.tirx.BufferStore(output, value, [i])

    with pytest.raises(Exception) as error:
        verify_body(source_module, make_body)
    message = str(error.value)
    assert "[VerifyParallelToPTO] control flow:" not in message
    assert {"branch": "conditional branches", "select": "Select", "predicate": "access predicate", "unsupported_call": "unsupported Call"}[
        kind
    ] in message


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "true_predicate", "scatter", "scalar_select_index", "uniform_select_load"])
def test_unconditional_indexed_access_remains_accepted(source_module, kind):
    def make_body(i, payload, index, output):
        if kind in ("gather", "true_predicate"):
            return tvm.tirx.BufferStore(output, payload[index[i]], [i], predicate=True if kind == "true_predicate" else None)
        if kind == "uniform_select_load":
            offset = index[i] + tvm.tirx.Select(index[0] > 0, index[0], index[1])
            return tvm.tirx.BufferStore(output, payload[i], [offset])
        offsets = index[i]
        if kind == "scalar_select_index":
            offsets = offsets + tvm.tirx.Select(index[0] > 0, 0, 1)
        return tvm.tirx.BufferStore(output, payload[i], [offsets])

    verify_body(source_module, make_body)


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather_after_branch", "scatter_after_branch", "saved_index_in_select"])
def test_conditional_context_does_not_spread_to_unconditional_accesses(source_module, kind):
    def make_body(i, payload, index, output):
        if kind == "saved_index_in_select":
            saved = tvm.tirx.Var("saved_index", "int32")
            offset = index[i] + tvm.tirx.Select(saved > 0, 0, 1)
            return tvm.tirx.SeqStmt([tvm.tirx.Bind(saved, index[index[i]]), tvm.tirx.BufferStore(output, payload[i], [offset])])
        branch = tvm.tirx.IfThenElse(i < 150, tvm.tirx.BufferStore(output, payload[i], [i]), None)
        access = (
            tvm.tirx.BufferStore(output, payload[index[i]], [i])
            if kind == "gather_after_branch"
            else tvm.tirx.BufferStore(output, payload[i], [index[i]])
        )
        return tvm.tirx.SeqStmt([branch, access])

    with pytest.raises(Exception) as error:
        verify_body(source_module, make_body)
    assert "[VerifyParallelToPTO] control flow:" not in str(error.value)
    assert ("unsupported indexed offset node" if kind == "saved_index_in_select" else "conditional branches") in str(error.value)


@pytest.mark.pto
def test_branch_local_index_alias_is_used_for_access_classification(source_module):
    def make_body(i, payload, index, output):
        saved = tvm.tirx.Var("saved_index", "int32")
        body = tvm.tirx.SeqStmt([tvm.tirx.Bind(saved, index[i]), tvm.tirx.BufferStore(output, payload[saved], [i])])
        return tvm.tirx.IfThenElse(i < 150, body, None)

    with pytest.raises(Exception, match="buffer `x` load.*requires gather.*IfThenElse then"):
        verify_body(source_module, make_body)


@pytest.mark.pto
def test_branch_definition_does_not_leak_into_else(source_module):
    def make_body(i, payload, index, output):
        alias = tvm.tirx.Var("then_only", "int32")
        return tvm.tirx.IfThenElse(
            i < 150,
            tvm.tirx.Bind(alias, i),
            tvm.tirx.BufferStore(output, payload[alias], [i]),
        )

    with pytest.raises(Exception, match="cannot be classified reliably.*then_only.*lexical scope.*IfThenElse else"):
        verify_body(source_module, make_body)


@pytest.mark.pto
def test_unanalyzable_conditional_address_reports_its_context(source_module):
    def make_body(i, payload, index, output):
        offset = tvm.tirx.call_extern("int32", "opaque_index", i)
        return tvm.tirx.IfThenElse(i < 150, tvm.tirx.BufferStore(output, payload[offset], [i]), None)

    with pytest.raises(Exception, match="buffer `x` load.*cannot be classified reliably.*unsupported call.*IfThenElse then"):
        verify_body(source_module, make_body)
