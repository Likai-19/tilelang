"""Host-only admission checks for direct PTO fill/clear.

These tests invoke the owning native Verify pass before scheduling or device
compilation. The full language pipeline and numerical writes are covered by
test_pto_fill_clear_run.py; nonzero starts and version-prefix boundaries are
covered by test_pto_parallel_vectorize_structure.py.
"""

from __future__ import annotations

import pytest

from tilelang import tvm
from tvm import tirx
import tilelang.ascend.language as T
from tilelang.ascend import transform
from tilelang.ascend.target import normalize_pto_target


def _verify(func):
    module = tvm.IRModule.from_expr(func.with_attr("target", normalize_pto_target("pto")))
    return transform.VerifyParallelToPTO()(module)


def _region_kernel(shape, extent, strides=None, dtype="float32", scope="shared.dyn", clear=False):
    @T.prim_func
    def main():
        owner = T.alloc_shared((1024,), dtype, scope=scope)
        b = T.StridedTensor(shape, strides, dtype, data=owner.data, scope=scope)
        with T.SimdVF(lanes=128):
            if len(shape) == 1:
                if clear:
                    T.clear(b[0:extent])
                else:
                    T.fill(b[0:extent], 3)
            else:
                if clear:
                    T.clear(b[0 : shape[0], 0:extent])
                else:
                    T.fill(b[0 : shape[0], 0:extent], 3)

    return main


@pytest.mark.pto
@pytest.mark.parametrize("clear", [False, True], ids=["fill", "clear"])
@pytest.mark.parametrize(
    "shape, extent, strides, scope, reason",
    [
        ((256,), 0, (1,), "shared.dyn", "positive compile-time integers"),
        ((256,), 1, (1,), "shared.dyn", "tail regions"),
        ((256,), 64, (1,), "shared.dyn", "tail regions"),
        ((3, 256), 128, (258, 1), "shared.dyn", "32-byte aligned"),
        ((3, 128), 128, (256, 2), "shared.dyn", "unit-stride axis"),
        ((128,), 128, (1,), "local.fragment", "only UB"),
    ],
    ids=["empty", "single-element", "short-vector", "unaligned-row", "no-contiguous-axis", "fragment"],
)
def test_fill_clear_verify_rejects_unsupported_regions(shape, extent, strides, scope, reason, clear):
    func = _region_kernel(shape, extent, strides, scope=scope, clear=clear)
    with pytest.raises(tvm.error.InternalError) as caught:
        _verify(func)
    message = str(caught.value)
    assert "[VerifyParallelToPTO]" in message
    assert reason in message


@pytest.mark.pto
@pytest.mark.parametrize("clear", [False, True], ids=["fill", "clear"])
@pytest.mark.parametrize("scope", ["shared", "shared.dyn"])
def test_fill_clear_verify_accepts_aligned_origin_zero_neighbor(clear, scope):
    func = _region_kernel((3, 256), 128, (256, 1), scope=scope, clear=clear)
    result = _verify(func)
    expected = tvm.IRModule.from_expr(func.with_attr("target", normalize_pto_target("pto")))
    tvm.ir.assert_structural_equal(result, expected)


@pytest.mark.pto
def test_fill_verify_rejects_runtime_region_extent():
    @T.prim_func
    def main(length: T.int32):
        b = T.alloc_shared((256,), "float32")
        with T.SimdVF(lanes=128):
            T.fill(b[0:length], T.float32(1))

    with pytest.raises(tvm.error.InternalError, match="positive compile-time integers"):
        _verify(main)


@pytest.mark.pto
def test_fill_verify_rejects_runtime_float_expression():
    @T.prim_func
    def main(value: T.float32):
        b = T.alloc_shared((128,), "float32")
        with T.SimdVF(lanes=128):
            T.fill(b, value + T.float32(1))

    with pytest.raises(tvm.error.InternalError, match="fill value must be a constant, uniform scalar"):
        _verify(main)


@pytest.mark.pto
@pytest.mark.parametrize("intermediate", ["float16", "bfloat16"])
def test_fill_verify_rejects_scalar_cast_producer_inside_vf(intermediate):
    @T.prim_func
    def main():
        source = T.alloc_shared((1,), "float32")
        b = T.alloc_shared((128,), "float32")
        with T.SimdVF(lanes=128):
            snapshot = T.bind(T.Cast("float32", T.Cast(intermediate, source[0])))
            T.clear(b)
            T.fill(b, snapshot)

    with pytest.raises(tvm.error.InternalError, match="put the casts directly in T.fill"):
        _verify(main)


@pytest.mark.pto
def test_fill_verify_accepts_bound_scalar_parameter_cast():
    @T.prim_func
    def main(value: T.float32):
        b = T.alloc_shared((128,), "float16")
        with T.SimdVF(lanes=128):
            snapshot = T.bind(T.Cast("float16", value))
            T.fill(b, snapshot)

    expected = tvm.IRModule.from_expr(main.with_attr("target", normalize_pto_target("pto")))
    tvm.ir.assert_structural_equal(_verify(main), expected)


@pytest.mark.pto
@pytest.mark.parametrize("launch_form", ["thread-binding", "thread-extent"])
@pytest.mark.parametrize("reverse", [False, True], ids=["block-index", "reversed-block-index"])
@pytest.mark.parametrize("offset", [0, -1, 2], ids=["in-bounds", "below-bound", "above-bound"])
def test_fill_verify_uses_launch_extent_for_scalar_indices(reverse, offset, launch_form):
    @T.prim_func
    def main(A: T.Tensor((2,), "float32"), B: T.Tensor((2, 128), "float32")):
        with T.Kernel(2) as bx:
            values = T.alloc_shared((2,), "float32")
            b = T.alloc_shared((128,), "float32")
            T.copy(A, values)
            with T.SimdVF(lanes=128):
                if reverse:
                    T.fill(b, values[1 - bx + offset])
                else:
                    T.fill(b, values[bx + offset])
            T.copy(b, B[bx, :])

    func = main
    if launch_form == "thread-extent":

        def rewrite(stmt):
            if isinstance(stmt, tirx.For) and stmt.kind == tirx.ForKind.THREAD_BINDING:
                binding = tirx.IterVar(
                    tvm.ir.Range.from_min_extent(stmt.min, stmt.extent),
                    stmt.loop_var,
                    tirx.IterVar.ThreadIndex,
                    stmt.thread_binding.thread_tag,
                )
                return tirx.AttrStmt(binding, "thread_extent", stmt.extent, stmt.body)
            return None

        func = main.with_body(tirx.stmt_functor.ir_transform(main.body, None, rewrite, ["tirx.For"]))
    if offset == 0:
        expected = tvm.IRModule.from_expr(func.with_attr("target", normalize_pto_target("pto")))
        tvm.ir.assert_structural_equal(_verify(func), expected)
    else:
        with pytest.raises(tvm.error.InternalError, match="scalar fill-value indices must be pure and provably in bounds"):
            _verify(func)


@pytest.mark.pto
@pytest.mark.parametrize("first_extent", [1, 4])
def test_fill_verify_restores_sibling_launch_bounds(first_extent):
    bx = tirx.Var("bx", "int32")
    values = tirx.decl_buffer((2,), "float32", "values", scope="shared.dyn")
    destination = tirx.decl_buffer((128,), "float32", "b", scope="shared.dyn")
    from testing.ascend._ir import region

    call = tirx.Call("handle", tvm.ir.Op.get("tl.tileop.fill"), [region(destination, 2), tirx.BufferLoad(values, [bx])])
    block = tirx.SBlock([], [], [], "SIMD_VF", tirx.Evaluate(call), annotations={"tl.simdvf_lanes": tirx.IntImm("int64", 128)})
    # Both launch attrs reuse the same binding identity. Their actual extents
    # control the proof; a stale first range must not constrain the sibling.
    binding = tirx.IterVar(tvm.ir.Range.from_min_extent(0, 1), bx, tirx.IterVar.ThreadIndex, "blockIdx.x")
    body = tirx.SeqStmt(
        [
            tirx.DeclBuffer(values),
            tirx.DeclBuffer(destination),
            tirx.AttrStmt(binding, "thread_extent", first_extent, tirx.Evaluate(0)),
            tirx.AttrStmt(binding, "thread_extent", 2, block),
        ]
    )
    func = tirx.PrimFunc([values.data, destination.data], body)
    expected = tvm.IRModule.from_expr(func.with_attr("target", normalize_pto_target("pto")))
    tvm.ir.assert_structural_equal(_verify(func), expected)


@pytest.mark.pto
@pytest.mark.parametrize("wrapper", ["if", "parallel"])
def test_fill_verify_rejects_nonuniform_statement_position(wrapper):
    @T.prim_func
    def main(condition: T.int32):
        b = T.alloc_shared((128,), "float32")
        with T.SimdVF(lanes=128):
            if wrapper == "if":
                if condition != 0:
                    T.fill(b, T.float32(1))
            else:
                for _i in T.Parallel(1):
                    T.fill(b, T.float32(1))

    with pytest.raises(tvm.error.InternalError, match="require a region-level statement"):
        _verify(main)


@pytest.mark.pto
@pytest.mark.parametrize("dtype", ["float64", "int64"])
def test_fill_verify_rejects_unsupported_storage_dtype(dtype):
    func = _region_kernel((128,), 128, (1,), dtype=dtype)
    with pytest.raises(tvm.error.InternalError, match="element dtype"):
        _verify(func)
