"""Host-only contracts for high-level direct reductions in PTO SIMD regions.

These tests feed explicit TIR to the native Verify/Vectorize passes. They
exercise support boundaries before codegen, including programs the Python
frontend normally rejects earlier. Device correctness lives in
test_pto_direct_reduce_run.py.
"""

from __future__ import annotations

import pytest

from tilelang import tvm
from tilelang.ascend import transform as ascend
from tilelang.ascend.target import normalize_pto_target
from tvm import tirx

from testing.ascend._ir import calls, nodes, region


def _module(
    shape=(3, 256),
    dim=1,
    dtype="float32",
    out_dtype=None,
    out_shape=None,
    strides=None,
    source_region=None,
    destination_region=None,
    shared_data=False,
    scope="shared",
    operation="sum",
    clear=True,
    lanes=128,
    annotations=None,
    wrapper=None,
):
    out_dtype = out_dtype or dtype
    out_shape = out_shape if out_shape is not None else ((1,) if len(shape) == 1 else (shape[1 - dim],))
    source = tirx.decl_buffer(shape, dtype, "a", scope=scope, strides=strides)
    destination = tirx.decl_buffer(out_shape, out_dtype, "b", scope="shared", data=source.data if shared_data else None)
    source_mins, source_extents = source_region or ([0] * len(shape), shape)
    destination_mins, destination_extents = destination_region or ([0] * len(out_shape), out_shape)
    call = tirx.Call(
        "handle",
        tvm.ir.Op.get("tl.tileop.reduce"),
        [
            region(source, 1, source_mins, source_extents),
            region(destination, 2, destination_mins, destination_extents),
            tirx.StringImm(operation),
            tirx.IntImm("int32", dim),
            tirx.IntImm("bool", clear),
        ],
        annotations=annotations or {},
    )
    body = tirx.Evaluate(call)
    if wrapper == "if":
        body = tirx.IfThenElse(tirx.IntImm("bool", True), body, None)
    elif wrapper == "while":
        body = tirx.While(tirx.IntImm("bool", False), body)
    elif wrapper == "parallel":
        body = tirx.For(tirx.Var("i", "int32"), 0, 2, tirx.ForKind.PARALLEL, body)
    elif wrapper == "serial":
        body = tirx.For(tirx.Var("i", "int32"), 0, 2, tirx.ForKind.SERIAL, body)
    elif wrapper == "unrolled":
        body = tirx.For(tirx.Var("i", "int32"), 0, 2, tirx.ForKind.UNROLLED, body)
    attributes = {} if lanes is None else {"tl.simdvf_lanes": tirx.IntImm("int64", lanes)}
    block = tirx.SBlock([], [], [], "SIMD_VF", body, annotations=attributes)
    declarations = [tirx.DeclBuffer(source), tirx.DeclBuffer(destination), block]
    parameters = [source.data] if shared_data else [source.data, destination.data]
    parameters += [extent for extent in shape if isinstance(extent, tirx.Var)]
    func = tirx.PrimFunc(parameters, tirx.SeqStmt(declarations)).with_attr("target", normalize_pto_target("pto"))
    return tvm.IRModule({"main": func})


def _tileop_expression_module(operation, position):
    module = _module(shape=(128,), dim=0, out_shape=(1,) if operation == "sum" else (128,))

    def rewrite(stmt):
        if not isinstance(stmt, tirx.Evaluate) or not isinstance(stmt.value, tirx.Call):
            return None
        call = stmt.value
        destination = call.args[1].args[0].buffer
        if operation != "sum":
            call = tirx.Call(
                "handle",
                tvm.ir.Op.get("tl.tileop.fill"),
                [call.args[1], tirx.FloatImm("float32", 0 if operation == "clear" else 3)],
            )
        if position == "bind":
            return tirx.Bind(tirx.Var("result", "handle"), call)
        if position == "call-argument":
            return tirx.Evaluate(tirx.call_extern("int32", "consume_tileop", call))
        if position == "call-annotation":
            return tirx.Evaluate(
                tirx.Call("int32", tvm.ir.Op.get("tirx.call_extern"), [tirx.StringImm("consume_tileop")], {"tileop": call})
            )
        if position == "tileop-annotation":
            return tirx.Evaluate(tirx.Call(call.dtype, call.op, call.args, {"tileop": call}))
        if position == "attr-value":
            return tirx.AttrStmt(tirx.IntImm("int32", 0), "tileop_metadata", call, tirx.Evaluate(0))
        if position == "attr-node":
            return tirx.AttrStmt(call, "tileop_metadata", 0, tirx.Evaluate(0))
        if position == "store-value":
            return tirx.BufferStore(destination, tirx.call_extern("float32", "consume_tileop", call), [0])
        if position in ("store-predicate", "load-predicate"):
            predicate = tirx.call_extern("bool", "consume_tileop", call)
            if position == "store-predicate":
                return tirx.BufferStore(destination, tirx.FloatImm("float32", 0), [0], predicate)
            return tirx.Evaluate(tirx.BufferLoad(destination, [0], predicate))
        body = tirx.Evaluate(call)
        if position in ("serial", "unrolled"):
            kind = tirx.ForKind.SERIAL if position == "serial" else tirx.ForKind.UNROLLED
            body = tirx.For(tirx.Var("i", "int32"), 0, 2, kind, body)
        return body

    return tvm.IRModule({"main": module["main"].with_body(tirx.stmt_functor.ir_transform(module["main"].body, None, rewrite))})


@pytest.mark.pto
@pytest.mark.parametrize("operation", ["sum", "fill", "clear"])
@pytest.mark.parametrize(
    "position",
    [
        "bind",
        "call-argument",
        "call-annotation",
        "tileop-annotation",
        "attr-value",
        "attr-node",
        "store-value",
        "store-predicate",
        "load-predicate",
    ],
)
def test_direct_tileop_verify_rejects_expression_positions(operation, position):
    module = _tileop_expression_module(operation, position)
    with pytest.raises(tvm.error.InternalError, match=r"\[VerifyParallelToPTO\].*direct Evaluate\(Call\)"):
        ascend.VerifyParallelToPTO()(module)


@pytest.mark.pto
@pytest.mark.parametrize("operation", ["sum", "fill", "clear"])
@pytest.mark.parametrize(
    "position", ["bind", "call-argument", "call-annotation", "attr-value", "attr-node", "store-value", "store-predicate", "load-predicate"]
)
def test_direct_tileop_vectorize_rejects_residual_calls_without_verify(operation, position):
    module = _tileop_expression_module(operation, position)
    with pytest.raises(
        tvm.error.InternalError,
        match=r"\[VectorizeParallelToPTO\].*must be standalone Evaluate\(Call\) statements",
    ):
        ascend.VectorizeParallelToPTO()(module)


@pytest.mark.pto
@pytest.mark.parametrize("operation", ["sum", "fill", "clear"])
@pytest.mark.parametrize("position", ["evaluate", "serial", "unrolled"])
def test_direct_tileop_accepts_standalone_evaluate(operation, position):
    module = _tileop_expression_module(operation, position)
    verified = ascend.VerifyParallelToPTO()(module)
    tvm.ir.assert_structural_equal(module, verified)
    lowered = ascend.VectorizeParallelToPTO()(verified)
    assert not calls(lowered["main"], "tl.tileop.reduce")
    assert not calls(lowered["main"], "tl.tileop.fill")
    assert calls(lowered["main"], "tl.vmi.vstore")


@pytest.mark.pto
@pytest.mark.parametrize(
    "shape, dim, lanes, out_shape, wrapper",
    [
        ((1,), 0, 64, (1,), None),
        ((259,), 0, 128, (1,), None),
        ((3, 256), 1, 64, (3,), None),
        ((3, 256), 0, 256, (1, 256), None),
        ((3, 130), 1, 128, (3, 1), "serial"),
        ((3, 130), 0, 128, (130,), "unrolled"),
    ],
    ids=["one-element", "1d-tail", "a-full", "b-full-keepdims", "a-tail-serial", "b-tail-unrolled"],
)
@pytest.mark.parametrize("clear", [True, False], ids=["clear", "accumulate"])
def test_direct_reduce_verify_preserves_accepted_ir(shape, dim, lanes, out_shape, wrapper, clear):
    module = _module(shape=shape, dim=dim, lanes=lanes, out_shape=out_shape, wrapper=wrapper, clear=clear)
    verified = ascend.VerifyParallelToPTO()(module)
    tvm.ir.assert_structural_equal(module, verified)
    lowered = ascend.VectorizeParallelToPTO()(verified)
    assert not calls(lowered["main"], "tl.tileop.reduce")
    assert calls(lowered["main"], "tl.vmi.vload")
    assert calls(lowered["main"], "tl.vmi.vstore")


@pytest.mark.pto
@pytest.mark.parametrize("operation, fold, cross", [("sum", "vadd", "vcadd"), ("max", "vmax", "vcmax"), ("min", "vmin", "vcmin")])
@pytest.mark.parametrize("dim", [0, 1], ids=["b", "a"])
def test_direct_reduce_selects_operation_and_exact_old_output_reads(operation, fold, cross, dim):
    module = _module(dim=dim, operation=operation, clear=False)
    verified = ascend.VerifyParallelToPTO()(module)
    lowered = ascend.VectorizeParallelToPTO()(verified)["main"]
    assert calls(lowered, "tl.vmi." + fold)
    assert bool(calls(lowered, "tl.vmi." + cross)) == (dim == 1)
    previous = [call for call in calls(lowered, "tl.vmi.vload") if call.args[0].args[0].buffer.name == "b"]
    assert len(previous) == 1
    assert int(previous[0].annotations["size"]) == (1 if dim == 1 else 128)
    assert int(previous[0].args[0].args[1]) == (1 if dim == 1 else 128)


@pytest.mark.pto
@pytest.mark.parametrize(
    "columns, lanes, dtype, out_dtype",
    [
        (3, 128, "float32", "float32"),
        (130, 128, "float32", "float32"),
        (257, 256, "int32", "int32"),
        (64, 64, "float16", "float16"),
        (128, 64, "float16", "float16"),
        (65, 64, "float16", "float32"),
    ],
    ids=["short", "tail", "vl256-tail", "narrow-read", "narrow-two-chunks", "widen-tail"],
)
@pytest.mark.parametrize("operation, fold", [("sum", "vadd"), ("max", "vmax"), ("min", "vmin")])
@pytest.mark.parametrize("clear", [True, False], ids=["clear", "accumulate"])
def test_direct_reduce_b_uses_one_masked_chunk_loop(columns, lanes, dtype, out_dtype, operation, fold, clear):
    module = _module(shape=(3, columns), dim=0, lanes=lanes, dtype=dtype, out_dtype=out_dtype, operation=operation, clear=clear)
    lowered = ascend.VectorizeParallelToPTO()(ascend.VerifyParallelToPTO()(module))["main"]
    loops = nodes(lowered, tirx.For)
    chunk_loops = [loop for loop in loops if loop.loop_var.name == "reduce_chunk"]
    assert len(chunk_loops) == 1
    assert int(chunk_loops[0].min) == 0
    assert int(chunk_loops[0].extent) == (columns + lanes - 1) // lanes
    # One chunk loop and one reduction-axis loop also cover the final tail;
    # there is no separate per-output scalar reduction.
    assert len(loops) == 2
    loads = calls(lowered, "tl.vmi.vload")
    assert all(int(call.annotations["size"]) == lanes for call in loads)
    for call in loads:
        physical_lanes = 2048 // call.dtype.bits
        assert int(call.args[0].args[1]) == (lanes + physical_lanes - 1) // physical_lanes * physical_lanes
    previous = [call for call in loads if call.args[0].args[0].buffer.name == "b"]
    assert len(previous) == int(not clear)
    masks = calls(lowered, "tl.vmi.create_mask")
    assert len(masks) == 1
    tvm.ir.assert_structural_equal(masks[0].args[0], columns - chunk_loops[0].loop_var * lanes)
    store = calls(lowered, "tl.vmi.vstore")
    assert len(store) == 1
    assert len(store[0].args) == 4
    folds = calls(lowered, "tl.vmi." + fold)
    assert len(folds) == 1 + int(not clear)
    assert all(call.args[2].same_as(store[0].args[3]) for call in folds)
    assert not calls(lowered, "tl.vmi.vsel")
    assert not calls(lowered, "tl.vmi.vcadd")


@pytest.mark.pto
@pytest.mark.parametrize(
    "extent, lanes, operation, fold",
    [
        (63, 128, "sum", "vadd"),
        (128, 128, "max", "vmax"),
        (129, 128, "min", "vmin"),
        (259, 128, "sum", "vadd"),
        (1024, 128, "sum", "vadd"),
        (1025, 128, "max", "vmax"),
        (64, 64, "min", "vmin"),
        (259, 256, "min", "vmin"),
        (1024, 256, "max", "vmax"),
    ],
    ids=["short", "one-full", "one-full-tail", "two-full-tail", "many-full", "many-full-tail", "vl64", "vl256-tail", "vl256-full"],
)
@pytest.mark.parametrize("clear", [True, False], ids=["clear", "accumulate"])
def test_direct_reduce_a_separates_full_chunks_from_tail(extent, lanes, operation, fold, clear):
    module = _module(shape=(3, extent), lanes=lanes, operation=operation, clear=clear)
    lowered = ascend.VectorizeParallelToPTO()(ascend.VerifyParallelToPTO()(module))["main"]
    full_chunks, tail = divmod(extent, lanes)
    has_tail_update = full_chunks > 0 and tail > 0
    mask_bindings = [
        bind for bind in nodes(lowered, tirx.Bind) if isinstance(bind.value, tirx.Call) and bind.value.op.name == "tl.vmi.create_mask"
    ]
    # Static shapes need only constant masks. A short first/only load is
    # restricted by the final cross-lane domain, rather than a tail update.
    assert all(isinstance(bind.value.args[0], tirx.IntImm) for bind in mask_bindings)

    def active_lanes(mask):
        return int(next(bind.value.args[0] for bind in mask_bindings if bind.var.same_as(mask)))

    folds = [call for call in calls(lowered, "tl.vmi." + fold) if call.dtype.lanes == lanes]
    assert sorted(active_lanes(call.args[2]) for call in folds) == sorted(
        ([lanes] if full_chunks > 1 else []) + ([tail] if has_tail_update else [])
    )
    selections = calls(lowered, "tl.vmi.vsel")
    assert len(selections) == int(has_tail_update)
    if selections:
        assert active_lanes(selections[0].args[0]) == tail
        tail_loads = [
            call
            for call in calls(lowered, "tl.vmi.vload")
            if call.args[0].args[0].buffer.name == "a"
            and isinstance(call.args[0].args[0].indices[1], tirx.IntImm)
            and int(call.args[0].args[0].indices[1]) == full_chunks * lanes
        ]
        assert len(tail_loads) == 1

    # The complete-chunk loop starts after the first loaded chunk. Its
    # body must not carry tail masks/selections, even when a tail follows.
    full_loops = [loop for loop in nodes(lowered, tirx.For) if int(loop.min) == 1]
    assert len(full_loops) == int(full_chunks > 1)
    if full_loops:
        assert int(full_loops[0].extent) == full_chunks - 1
        assert not calls(full_loops[0].body, "tl.vmi.create_mask")
        assert not calls(full_loops[0].body, "tl.vmi.vsel")
    previous = [call for call in calls(lowered, "tl.vmi.vload") if call.args[0].args[0].buffer.name == "b"]
    assert len(previous) == int(not clear)
    assert all(int(call.annotations["size"]) == 1 for call in previous)


@pytest.mark.pto
@pytest.mark.parametrize(
    "options, reason",
    [
        ({"source_region": ((0, 0), (3, 128))}, "complete logical"),
        ({"source_region": ((0, 8), (3, 128))}, "complete logical"),
        ({"destination_region": ((0,), (1,))}, "complete logical"),
        ({"out_shape": (2,)}, "destination shape"),
        ({"out_shape": (3, 2)}, "keepdims destination shape"),
        ({"shape": (256,), "dim": 0, "out_shape": (2,)}, "destination shape"),
        ({"shared_data": True}, "overlapping source/destination"),
        ({"dtype": "bfloat16"}, "dtype combination"),
        ({"dtype": "int16"}, "dtype combination"),
        ({"strides": (512, 2)}, "non-overlapping rows or columns"),
        ({"operation": "prod"}, "only sum"),
        ({"annotations": {"batch": tirx.IntImm("int32", 2)}}, "batch=1"),
        ({"annotations": {"nan_propagate": tirx.IntImm("bool", True)}}, "nan_propagate=False"),
        ({"shape": (3, tirx.Var("n", "int32"))}, "positive compile-time"),
        ({"shape": (2, 3, 256), "dim": 2, "out_shape": (2, 3)}, "logical 1D/2D"),
        ({"scope": "global"}, "stage through UB"),
        ({"lanes": None}, "tl.simdvf_lanes"),
        ({"lanes": 32}, "one of"),
        ({"wrapper": "if"}, "region-level"),
        ({"wrapper": "while"}, "region-level"),
        ({"wrapper": "parallel"}, "region-level"),
    ],
    ids=[
        "source-prefix",
        "source-offset-region",
        "destination-prefix",
        "wrong-output-shape",
        "wrong-keepdims-shape",
        "wrong-1d-output",
        "shared-storage",
        "bf16-accumulator",
        "int16-accumulator",
        "no-unit-stride",
        "unsupported-operation",
        "batch",
        "nan-mode",
        "dynamic-extent",
        "rank3",
        "global",
        "missing-lanes",
        "invalid-lanes",
        "conditional-tileop",
        "while-tileop",
        "parallel-tileop",
    ],
)
def test_direct_reduce_verify_rejects_unsupported_contract(options, reason):
    module = _module(**options)
    with pytest.raises(tvm.error.InternalError, match=r"\[VerifyParallelToPTO\]") as error:
        ascend.VerifyParallelToPTO()(module)
    assert reason in str(error.value)


@pytest.mark.pto
@pytest.mark.parametrize(
    "dtype, accumulator",
    [
        ("float32", "float32"),
        ("float16", "float16"),
        ("float16", "float32"),
        ("bfloat16", "float32"),
        ("int32", "int32"),
        ("uint32", "uint32"),
    ],
)
@pytest.mark.parametrize("dim", [0, 1], ids=["b", "a"])
def test_direct_reduce_supported_dtype_routes(dtype, accumulator, dim):
    module = _module(dtype=dtype, out_dtype=accumulator, dim=dim)
    lowered = ascend.VectorizeParallelToPTO()(ascend.VerifyParallelToPTO()(module))["main"]
    loads = calls(lowered, "tl.vmi.vload")
    assert all(call.dtype == dtype + "x128" for call in loads)
    conversions = calls(lowered, "tl.vmi.vcvt")
    assert bool(conversions) == (dtype != accumulator)
    assert all(call.dtype == accumulator + "x128" for call in conversions)


@pytest.mark.pto
@pytest.mark.parametrize("operation", ["abssum", "absmax"])
@pytest.mark.parametrize(
    "dtype, accumulator",
    [
        ("float32", "float32"),
        ("float16", "float16"),
        ("float16", "float32"),
        ("bfloat16", "float32"),
        ("int32", "int32"),
        ("uint32", "uint32"),
    ],
)
@pytest.mark.parametrize("dim", [0, 1], ids=["b", "a"])
@pytest.mark.parametrize("clear", [True, False], ids=["clear", "accumulate"])
def test_direct_reduce_absolute_dtype_routes(operation, dtype, accumulator, dim, clear):
    module = _module(operation=operation, dtype=dtype, out_dtype=accumulator, dim=dim, clear=clear)
    verified = ascend.VerifyParallelToPTO()(module)
    tvm.ir.assert_structural_equal(module, verified)
    lowered = ascend.VectorizeParallelToPTO()(verified)["main"]
    assert not calls(lowered, "tl.tileop.reduce")
    absolute = calls(lowered, "tl.vmi.vabs")
    assert bool(absolute) == (dtype != "uint32")
    assert all(call.dtype == accumulator + "x128" for call in absolute)
    assert bool(calls(lowered, "tl.vmi.vcvt")) == (dtype != accumulator)
    assert bool(calls(lowered, "tl.vmi.vcadd" if operation == "abssum" else "tl.vmi.vcmax")) == (dim == 1)
    assert not calls(lowered, "tl.vmi.vrelu")
    zeros = calls(lowered, "tl.vmi.vbrc")
    assert len(zeros) == int(operation == "absmax")
    assert all(call.dtype == accumulator + "x128" and float(call.args[0].value) == 0 for call in zeros)
