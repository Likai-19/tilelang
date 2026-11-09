"""PTO Parallel vectorization structure tests.

Hardware independent: each case lowers TileLang IR for the PTO target with
:func:`tilelang.engine.lower.lower` (TileLang IR only, no PTOAS/Bisheng device
compilation and no device execution) and inspects the module produced by
``VectorizeParallelToPTO`` through a capture wrapper.

Covered contract:

* a divisible unit reuses the shared full mask (``create_mask(L)``) and emits
  no ``remaining`` state;
* a non-divisible unit allocates a scalar ``remaining`` (scope ``local.var``),
  initializes it to the logical extent ``E`` once, binds a per-chunk
  ``create_mask(remaining)`` before the body and decrements it by ``L``;
* a 2D unit keeps the non-selected coordinate as an outer serial loop and
  resets ``remaining`` inside it, so rows are independent;
* the real buffer indices (including a padded row stride) survive
  vectorization: addresses keep the original buffers and stride expressions;
* the region emits ``vload`` / ``vstore`` / ``create_mask`` VMI calls with the
  chunk mask on the store;
* explicit padding (``T.Parallel(P)`` with an ``if`` guard) is currently
  rejected: guard legalization is left to a later change.
"""

from __future__ import annotations

from contextlib import contextmanager

import re

import pytest

import tilelang.ascend.language as T
from tilelang.ascend import transform as ascend_transform
from tilelang.ascend.target import normalize_pto_target
from tilelang.engine.lower import lower as _lower
import tvm

_REMAINING = "remaining"


@contextmanager
def _verify_boundary_capture():
    """Capture the module handed to VerifyParallelToPTO, then delegate.

    Self-contained copy of the contract suite's helper so this file runs
    standalone (cross-test-module imports are fragile under pytest's
    rootdir-dependent module resolution).
    """
    captured = {}
    original = ascend_transform.VerifyParallelToPTO

    def capturing_factory():
        verify_pass = original()

        def run(mod):
            captured["module"] = mod
            return verify_pass(mod)

        return run

    ascend_transform.VerifyParallelToPTO = capturing_factory
    try:
        yield captured
    finally:
        ascend_transform.VerifyParallelToPTO = original


def _ceil_to_lanes(extent: int, lanes: int) -> int:
    return (extent + lanes - 1) // lanes * lanes


def _as_int(expr) -> int:
    if isinstance(expr, int):
        return expr
    value = getattr(expr, "value", None)
    assert value is not None, f"expected an integer constant, got {expr!r}"
    return int(value)


@contextmanager
def _vectorize_capture():
    """Capture the module returned by VectorizeParallelToPTO, then delegate."""
    captured = {}
    original = ascend_transform.VectorizeParallelToPTO

    def capturing_factory():
        vectorize_pass = original()

        def run(mod):
            result = vectorize_pass(mod)
            captured["module"] = result
            return result

        return run

    ascend_transform.VectorizeParallelToPTO = capturing_factory
    try:
        yield captured
    finally:
        ascend_transform.VectorizeParallelToPTO = original


def _lower_capture(func, *, expect_error=False):
    """Lower *func* for PTO and return (module_after_vectorize, error)."""
    target = normalize_pto_target("pto")
    error = None
    with _vectorize_capture() as captured:
        try:
            with target:
                _lower(func, target=target)
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            error = exc
    if error is not None and not expect_error:
        raise error
    return captured.get("module"), error


def _body_of(mod):
    assert len(mod.functions) == 1, f"expected a single function, got {list(mod.functions)}"
    return next(iter(mod.functions.values())).body


def _collect(mod, predicate):
    found = []

    def visit(node):
        if predicate(node):
            found.append(node)

    tvm.tirx.stmt_functor.post_order_visit(_body_of(mod), visit)
    return found


def _vmi_calls(mod):
    calls = {}

    def visit(node):
        if isinstance(node, tvm.tirx.Call):
            op = getattr(node, "op", None)
            name = getattr(op, "name", None)
            if isinstance(name, str) and name.startswith("tl.vmi."):
                calls.setdefault(name, []).append(node)

    tvm.tirx.stmt_functor.post_order_visit(_body_of(mod), visit)
    return calls


def _buffers(mod):
    buffers = {}

    def visit(node):
        if isinstance(node, tvm.tirx.AllocBuffer):
            buffers[node.buffer.name] = node.buffer
        if isinstance(node, (tvm.tirx.BufferStore, tvm.tirx.BufferLoad)):
            buffers[node.buffer.name] = node.buffer

    tvm.tirx.stmt_functor.post_order_visit(_body_of(mod), visit)
    return buffers


def _remaining_buffers(mod):
    return [buf for name, buf in _buffers(mod).items() if name.startswith(_REMAINING)]


def _remaining_stores(mod, buffer):
    stores = []

    def visit(node):
        if isinstance(node, tvm.tirx.BufferStore) and node.buffer.same_as(buffer):
            stores.append(node)

    tvm.tirx.stmt_functor.post_order_visit(_body_of(mod), visit)
    return stores


def _subexpr_contains_constant(expr, value: int) -> bool:
    """Return whether *expr* contains an IntImm equal to *value*."""
    found = []

    def visit(node):
        if isinstance(node, tvm.tirx.IntImm) and int(node.value) == value:
            found.append(node)

    tvm.tirx.stmt_functor.post_order_visit(tvm.tirx.Evaluate(expr), visit)
    return bool(found)


@pytest.mark.pto
@pytest.mark.parametrize(
    "shape, dim, keepdims, source_versions, destination_versions",
    [
        ((256,), 0, False, 2, 2),
        ((3, 128), 0, False, 2, 2),
        ((3, 128), 1, True, 2, 2),
        ((1, 128), 1, True, 1, 1),
        ((3, 128), 1, False, 2, 1),
        ((3, 128), 0, True, 1, 2),
    ],
)
def test_direct_reduce_uses_existing_version_contract(shape, dim, keepdims, source_versions, destination_versions):
    out_shape = (
        tuple(1 if i == dim else extent for i, extent in enumerate(shape))
        if keepdims
        else tuple(extent for i, extent in enumerate(shape) if i != dim) or (1,)
    )

    @T.prim_func
    def main(A: T.Tensor((4, *shape), "float32"), B: T.Tensor((4, *out_shape), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared(shape, "float32")
            b = T.alloc_shared(out_shape, "float32")
            T.annotate_buffer_versions({a: source_versions, b: destination_versions})
            for t in T.Pipelined(4, num_stages=2):
                if len(shape) == 1:
                    T.copy(A[t, :], a)
                else:
                    T.copy(A[t, :, :], a)
                with T.SimdVF(lanes=128):
                    T.reduce_sum(a, b, dim=dim)
                if len(out_shape) == 1:
                    T.copy(b, B[t, :])
                else:
                    T.copy(b, B[t, :, :])

    mod, error = _lower_capture(main)
    assert error is None
    operations = _vmi_calls(mod)
    assert "tl.vmi.vload" in operations and "tl.vmi.vstore" in operations
    assert not _collect(mod, lambda node: isinstance(node, tvm.tirx.Call) and getattr(node.op, "name", "") == "tl.tileop.reduce")
    assert ("tl.vmi.vcadd" in operations) == (dim == len(shape) - 1)
    for operation, versions in (("tl.vmi.vload", source_versions), ("tl.vmi.vstore", destination_versions)):
        address_arg = 0 if operation.endswith("vload") else 1
        for call in operations[operation]:
            load = call.args[address_arg].args[0]
            logical = shape if address_arg == 0 else out_shape
            assert tuple(int(extent) for extent in load.buffer.shape) == ((versions, *logical) if versions > 1 else logical)


@pytest.mark.pto
@pytest.mark.parametrize("operation", ["fill", "clear", "reduce"])
@pytest.mark.parametrize("inner_lanes", [64, 128], ids=["same-lanes", "different-lanes"])
def test_tileops_reject_nested_simd_vf_in_verify(operation, inner_lanes):
    @T.prim_func
    def main(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
        with T.Kernel(1):
            b = T.alloc_shared((64,), "float32")
            result = T.alloc_shared((1,), "float32")
            T.copy(A, b)
            with T.SimdVF(lanes=64), T.SimdVF(lanes=inner_lanes):
                if operation == "fill":
                    T.fill(b, T.float32(7))
                elif operation == "clear":
                    T.clear(b)
                else:
                    T.reduce_sum(b, result, dim=0)
            T.copy(b, B)

    with _verify_boundary_capture() as captured:
        mod, error = _lower_capture(main, expect_error=True)
    assert "module" in captured
    assert error is not None and mod is None
    message = str(error)
    assert "[VerifyParallelToPTO]" in message and "nested SIMD_VF" in message


@pytest.mark.pto
def test_tileops_accept_sequential_simd_vf_with_independent_lanes():
    @T.prim_func
    def main(B: T.Tensor((128,), "float32")):
        with T.Kernel(1):
            b = T.alloc_shared((128,), "float32")
            with T.SimdVF(lanes=64):
                T.fill(b, T.float32(7))
            with T.SimdVF(lanes=128):
                T.clear(b)
            T.copy(b, B)

    mod, error = _lower_capture(main)
    assert error is None
    operations = _vmi_calls(mod)
    assert sorted(int(call.annotations["size"]) for call in operations["tl.vmi.vbrc"]) == [64, 128]
    assert sorted(int(call.args[1].args[1]) for call in operations["tl.vmi.vstore"]) == [64, 128]


@pytest.mark.pto
@pytest.mark.parametrize("rows, columns", [(3, 256), (2, 128)])
def test_fill_clear_keeps_origin_zero_region_and_value_snapshot(rows, columns):
    @T.prim_func
    def main(A: T.Tensor((3, 256), "float32"), B: T.Tensor((3, 256), "float32")):
        with T.Kernel(1):
            b = T.alloc_shared((3, 256), "float32")
            T.copy(A, b)
            with T.SimdVF(lanes=64):
                T.fill(b[0:rows, 0:columns], b[1, 70])
                T.clear(b[0:1, 0:64])
            T.copy(b, B)

    mod, error = _lower_capture(main)
    assert error is None
    operations = _vmi_calls(mod)
    assert "tl.vmi.vstore" in operations and "tl.vmi.vbrc" in operations
    assert len(operations["tl.vmi.vload"]) == 1
    snapshot = operations["tl.vmi.vload"][0]
    assert int(snapshot.annotations["size"]) == 1
    assert int(snapshot.args[0].args[1]) == 1
    assert "tl.vmi.vsel" not in operations
    assert not _collect(mod, lambda node: isinstance(node, tvm.tirx.Call) and getattr(node.op, "name", "") == "tl.tileop.fill")
    # The scalar snapshot is outside both generated row/chunk loops. The
    # source contains one value read, even when it aliases the written region.
    assert not _collect(
        mod,
        lambda node: (
            isinstance(node, tvm.tirx.For)
            and bool(_collect(tvm.IRModule({"main": tvm.tirx.PrimFunc([], node.body)}), lambda n: n.same_as(snapshot)))
        ),
    )


@pytest.mark.pto
@pytest.mark.parametrize("extent, length, lanes", [(64, 64, 64), (256, 128, 64), (256, 256, 128), (512, 512, 256)])
@pytest.mark.parametrize("clear", [False, True], ids=["fill", "clear"])
def test_fill_clear_materializes_full_vectors(extent, length, lanes, clear):
    @T.prim_func
    def main(A: T.Tensor((extent,), "float32"), B: T.Tensor((extent,), "float32")):
        with T.Kernel(1):
            b = T.alloc_shared((extent,), "float32")
            T.copy(A, b)
            with T.SimdVF(lanes=lanes):
                if clear:
                    T.clear(b[0:length])
                else:
                    T.fill(b[0:length], T.float32(3.25))
            T.copy(b, B)

    mod, error = _lower_capture(main)
    assert error is None
    operations = _vmi_calls(mod)
    assert [int(call.annotations["size"]) for call in operations["tl.vmi.vbrc"]] == [lanes]
    assert "tl.vmi.vand" not in operations and "tl.vmi.vnot" not in operations
    assert all(int(call.args[0]) == lanes for call in operations["tl.vmi.create_mask"])
    for store in operations["tl.vmi.vstore"]:
        assert int(store.args[0].dtype.lanes) == lanes
        assert int(store.args[1].args[1]) == lanes
    assert not _remaining_buffers(mod)


@pytest.mark.pto
@pytest.mark.parametrize("intermediate", ["float16", "bfloat16"])
def test_fill_nested_float_casts_use_full_width_vector_conversions(intermediate):
    @T.prim_func
    def main(A: T.Tensor((256,), "float32"), B: T.Tensor((256,), "float32")):
        with T.Kernel(1):
            b = T.alloc_shared((256,), "float32")
            T.copy(A, b)
            with T.SimdVF(lanes=128):
                T.fill(b, T.Cast("float32", T.Cast(intermediate, b[70])))
            T.copy(b, B)

    mod, error = _lower_capture(main)
    assert error is None
    operations = _vmi_calls(mod)
    assert [int(call.annotations["size"]) for call in operations["tl.vmi.vload"]] == [1]
    assert [int(call.annotations["size"]) for call in operations["tl.vmi.vbrc"]] == [128]
    conversions = operations["tl.vmi.vcvt"]
    assert [str(call.dtype) for call in conversions] == [intermediate + "x128", "float32x128"]
    assert (
        conversions[1]
        .args[0]
        .same_as(
            next(bind.var for bind in _collect(mod, lambda node: isinstance(node, tvm.tirx.Bind)) if bind.value.same_as(conversions[0]))
        )
    )


@pytest.mark.pto
@pytest.mark.parametrize(
    "intermediate, dtype, bind_cast",
    [
        ("float16", "float16", False),
        ("float16", "float32", False),
        ("float16", "float16", True),
        ("float16", "float32", True),
        ("bfloat16", "float32", False),
    ],
    ids=["direct-fp16", "direct-chain", "bound-fp16", "bound-chain", "bf16-vector-control"],
)
def test_fill_parameter_cast_preserves_scalar_expression(intermediate, dtype, bind_cast):
    @T.prim_func
    def main(value: T.float32, B: T.Tensor((3, 256), dtype)):
        with T.Kernel(1):
            b = T.alloc_shared((3, 256), dtype)
            with T.SimdVF(lanes=128):
                if bind_cast:
                    snapshot = T.bind(T.Cast(dtype, T.Cast(intermediate, value)))
                    T.fill(b, snapshot)
                else:
                    T.fill(b, T.Cast(dtype, T.Cast(intermediate, value)))
            T.copy(b, B)

    mod, error = _lower_capture(main)
    assert error is None
    operations = _vmi_calls(mod)
    vf = _collect(mod, lambda node: isinstance(node, tvm.tirx.SBlock) and node.name_hint == "SIMD_VF")[0]
    casts = _collect(mod, lambda node: isinstance(node, tvm.tirx.Cast) and str(node.dtype) == intermediate)
    if intermediate == "float16":
        assert "tl.vmi.vcvt" not in operations
        assert len(casts) == 1
        # Scalar casts remain the broadcast argument in the VF, just as on
        # AscendC. Lowering must not move them to a separate VF prelude.
        assert _collect(tvm.IRModule({"main": tvm.tirx.PrimFunc([], vf.body)}), lambda node: node.same_as(casts[0]))
        broadcast = operations["tl.vmi.vbrc"][0]
        value = next(param for param in next(iter(mod.functions.values())).params if param.name == "value")
        expected = tvm.tirx.Cast(intermediate, value)
        if dtype != intermediate:
            expected = tvm.tirx.Cast(dtype, expected)
        bindings = {node.var: node.value for node in _collect(mod, lambda node: isinstance(node, tvm.tirx.Bind))}
        argument = broadcast.args[0]
        while isinstance(argument, tvm.tirx.Var) and argument in bindings:
            argument = bindings[argument]
        tvm.ir.assert_structural_equal(argument, expected)
    else:
        assert [str(call.dtype) for call in operations["tl.vmi.vcvt"]] == ["bfloat16x128", "float32x128"]
    assert [int(call.annotations["size"]) for call in operations["tl.vmi.vbrc"]] == [128]


@pytest.mark.pto
@pytest.mark.parametrize("clear", [False, True], ids=["fill", "clear"])
@pytest.mark.parametrize(
    "shape, starts, extents",
    [
        ((256,), (8,), (128,)),
        ((3, 256), (1, 0), (2, 128)),
        ((3, 256), (0, 8), (3, 128)),
        ((256,), (0,), (130,)),
        ((3, 256), (0, 0), (3, 130)),
    ],
    ids=["1d-nonzero-start", "2d-outer-nonzero-start", "2d-inner-nonzero-start", "1d-tail", "2d-tail"],
)
def test_fill_clear_rejects_nonzero_starts_and_tails_in_verify(shape, starts, extents, clear):
    @T.prim_func
    def main(A: T.Tensor(shape, "float32"), B: T.Tensor(shape, "float32")):
        with T.Kernel(1):
            b = T.alloc_shared(shape, "float32")
            T.copy(A, b)
            with T.SimdVF(lanes=128):
                if len(shape) == 1:
                    if clear:
                        T.clear(b[starts[0] : starts[0] + extents[0]])
                    else:
                        T.fill(b[starts[0] : starts[0] + extents[0]], 3.25)
                else:
                    if clear:
                        T.clear(b[starts[0] : starts[0] + extents[0], starts[1] : starts[1] + extents[1]])
                    else:
                        T.fill(b[starts[0] : starts[0] + extents[0], starts[1] : starts[1] + extents[1]], 3.25)
            T.copy(b, B)

    with _verify_boundary_capture() as captured:
        mod, error = _lower_capture(main, expect_error=True)
    assert "module" in captured, "the invalid fill/clear must reach VerifyParallelToPTO"
    assert error is not None
    assert mod is None, "Verify must reject the unsupported region before VectorizeParallelToPTO"
    message = str(error)
    assert "VerifyParallelToPTO" in message and "fill/clear" in message
    assert ("nonzero" if any(starts) else "tail") in message


@pytest.mark.pto
@pytest.mark.parametrize("offset", [0, 8, 32])
@pytest.mark.parametrize("clear", [False, True], ids=["fill", "clear"])
def test_fill_clear_rejects_nonzero_destination_storage_offset(offset, clear):
    @T.prim_func
    def main(A: T.Tensor((256,), "float32"), B: T.Tensor((256,), "float32")):
        with T.Kernel(1):
            owner = T.alloc_shared((256,), "float32")
            view = tvm.tirx.decl_buffer((128,), "float32", data=owner.data, scope="shared.dyn", elem_offset=offset)
            T.copy(A, owner)
            with T.SimdVF(lanes=128):
                if clear:
                    T.clear(view)
                else:
                    T.fill(view, 3.25)
            T.copy(owner, B)

    with _verify_boundary_capture() as captured:
        mod, error = _lower_capture(main, expect_error=offset != 0)
    assert "module" in captured
    if offset == 0:
        assert error is None
        assert "tl.vmi.vstore" in _vmi_calls(mod)
    else:
        assert error is not None
        assert mod is None
        message = str(error)
        assert "VerifyParallelToPTO" in message and "fill/clear" in message
        assert "nonzero destination storage offsets" in message


@pytest.mark.pto
@pytest.mark.parametrize("start, extent", [(1, 128), (0, 130)], ids=["nonzero-logical-start", "logical-tail"])
@pytest.mark.parametrize("clear", [False, True], ids=["fill", "clear"])
def test_fill_clear_version_axis_does_not_hide_logical_boundaries(start, extent, clear):
    @T.prim_func
    def main(A: T.Tensor((4, 3, 256), "float32"), B: T.Tensor((4, 3, 256), "float32")):
        with T.Kernel(1):
            b = T.alloc_shared((3, 256), "float32")
            T.annotate_buffer_versions({b: 2})
            for t in T.Pipelined(4, num_stages=2):
                T.copy(A[t, :, :], b)
                with T.SimdVF(lanes=128):
                    if clear:
                        T.clear(b[start:3, 0:extent])
                    else:
                        T.fill(b[start:3, 0:extent], 3.25)
                T.copy(b, B[t, :, :])

    with _verify_boundary_capture() as captured:
        mod, error = _lower_capture(main, expect_error=True)
    assert "module" in captured
    assert any(tuple(int(extent) for extent in buffer.shape) == (2, 3, 256) for buffer in _buffers(captured["module"]).values())
    assert error is not None
    assert mod is None
    message = str(error)
    assert "VerifyParallelToPTO" in message and "fill/clear" in message
    assert ("nonzero" if start else "tail") in message


@pytest.mark.pto
@pytest.mark.parametrize("versions", [1, 2])
@pytest.mark.parametrize("clear", [False, True], ids=["fill", "clear"])
def test_fill_clear_accepts_compiler_version_axis(versions, clear):
    @T.prim_func
    def main(A: T.Tensor((4, 3, 256), "float32"), B: T.Tensor((4, 3, 256), "float32")):
        with T.Kernel(1):
            b = T.alloc_shared((3, 256), "float32")
            T.annotate_buffer_versions({b: versions})
            for t in T.Pipelined(4, num_stages=2):
                T.copy(A[t, :, :], b)
                with T.SimdVF(lanes=128):
                    if clear:
                        T.clear(b)
                    else:
                        T.fill(b, b[0, 0])
                T.copy(b, B[t, :, :])

    mod, error = _lower_capture(main)
    assert error is None
    operations = _vmi_calls(mod)
    for store in operations["tl.vmi.vstore"]:
        address = store.args[1].args[0]
        assert tuple(int(extent) for extent in address.buffer.shape) == ((versions, 3, 256) if versions > 1 else (3, 256))
        assert int(store.args[1].args[1]) == 128


@pytest.mark.pto
@pytest.mark.parametrize("dim", [0, 1])
def test_reduce_clear_false_uses_template_read_width(dim):
    count = 3 if dim == 1 else 130

    @T.prim_func
    def main(A: T.Tensor((3, 256), "float32"), Initial: T.Tensor((count,), "float32"), B: T.Tensor((count,), "float32")):
        with T.Kernel(1):
            owner = T.alloc_shared((3, 256), "float32")
            a = T.StridedTensor((3, 130), (256, 1), "float32", data=owner.data, scope="shared.dyn")
            b = T.alloc_shared((count,), "float32")
            T.copy(A, owner)
            T.copy(Initial, b)
            with T.SimdVF(lanes=128):
                T.reduce_sum(a, b, dim=dim, clear=False)
            T.copy(b, B)

    mod, error = _lower_capture(main)
    assert error is None
    calls = _vmi_calls(mod)
    previous_reads = [call for call in calls["tl.vmi.vload"] if call.args[0].args[0].buffer.name == "b"]
    assert sorted(int(call.annotations["size"]) for call in previous_reads) == ([1] if dim == 1 else [128])
    for call in previous_reads:
        assert int(call.args[0].args[1]) == int(call.annotations["size"])
    assert ("tl.vmi.vcadd" in calls) == (dim == 1)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def _k1d(E, out_extent, lanes, dtype="float32"):
    P = _ceil_to_lanes(E, lanes)

    @T.prim_func
    def main(
        A: T.Tensor((P,), dtype),
        B_init: T.Tensor((out_extent,), dtype),
        B: T.Tensor((out_extent,), dtype),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((P,), dtype)
            b = T.alloc_shared((out_extent,), dtype)
            T.copy(A, a)
            T.copy(B_init, b)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    b[i] = a[i] + a[i]
            T.copy(b, B)

    return main


def _k2d_row(M, N, lanes, dtype="float32"):
    P = _ceil_to_lanes(N, lanes)

    @T.prim_func
    def main(
        A: T.Tensor((M * P,), dtype),
        B_init: T.Tensor((M * P,), dtype),
        B: T.Tensor((M * P,), dtype),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((M * P,), dtype)
            b = T.alloc_shared((M * P,), dtype)
            T.copy(A, a)
            T.copy(B_init, b)
            with T.SimdVF(lanes=lanes), T.Parallel(M, N) as (i, j):
                b[i * P + j] = a[i * P + j] + a[i * P + j]
            T.copy(b, B)

    return main


def _k_explicit_padding(E, lanes, dtype="float32"):
    """Explicit ``T.Parallel(P)`` with a user ``if`` guard: at this stage the
    guard is still rejected; its legalization belongs to the control-flow
    stage."""

    P = _ceil_to_lanes(E, lanes)

    @T.prim_func
    def main(
        A: T.Tensor((P,), dtype),
        B: T.Tensor((P,), dtype),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((P,), dtype)
            b = T.alloc_shared((P,), dtype)
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(P):
                    if i < E:
                        b[i] = a[i] + a[i]
                    else:
                        b[i] = a[i]
            T.copy(b, B)

    return main


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.pto
def test_divisible_unit_uses_the_shared_full_mask_without_remaining():
    E, lanes = 256, 128
    mod, _ = _lower_capture(_k1d(E, E, lanes))

    assert not _remaining_buffers(mod), "a divisible unit must not allocate remaining"
    calls = _vmi_calls(mod)
    assert "tl.vmi.create_mask" in calls, "the region must create the shared full mask"
    mask_sizes = [_as_int(call.args[0]) for call in calls["tl.vmi.create_mask"]]
    assert mask_sizes == [lanes], f"the shared mask must be created with L={lanes}, got {mask_sizes}"
    assert "tl.vmi.vload" in calls and "tl.vmi.vstore" in calls

    fors = _collect(mod, lambda n: isinstance(n, tvm.tirx.For))
    extents = sorted(_as_int(f.extent) for f in fors)
    assert extents == [2], f"only the Q=2 chunk loop may remain, got {extents}"


@pytest.mark.pto
def test_nondivisible_unit_builds_remaining_prefix_mask():
    E, lanes = 150, 128
    P = _ceil_to_lanes(E, lanes)
    mod, _ = _lower_capture(_k1d(E, P, lanes))

    buffers = _remaining_buffers(mod)
    assert len(buffers) == 1, f"expected exactly one remaining buffer, got {buffers}"
    remaining = buffers[0]
    assert remaining.scope() == "local.var", f"remaining must be a scalar local, got {remaining.scope()}"

    stores = _remaining_stores(mod, remaining)
    values = [_as_int(store.value) for store in stores if isinstance(store.value, tvm.tirx.IntImm)]
    assert E in values, f"remaining must be initialized to the logical extent {E}, got {values}"

    calls = _vmi_calls(mod)
    # The per-chunk mask size lives in the Call kwargs (`size`), which the
    # Python FFI does not expose; the observable contract is the value
    # argument: exactly one create_mask per unit whose source is the remaining
    # scalar load.
    prefix_masks = [
        call
        for call in calls.get("tl.vmi.create_mask", [])
        if isinstance(call.args[0], tvm.tirx.BufferLoad) and call.args[0].buffer.same_as(remaining)
    ]
    assert len(prefix_masks) == 1, f"expected one prefixed create_mask(remaining), got {len(prefix_masks)}"

    fors = _collect(mod, lambda n: isinstance(n, tvm.tirx.For))
    assert [_as_int(f.extent) for f in fors] == [2], f"expected the Q=2 chunk loop, got {fors}"


@pytest.mark.pto
def test_2d_unit_resets_remaining_inside_the_kept_loop():
    M, N, lanes = 4, 150, 128
    mod, _ = _lower_capture(_k2d_row(M, N, lanes))

    buffers = _remaining_buffers(mod)
    assert len(buffers) == 1, f"expected exactly one remaining buffer, got {buffers}"
    remaining = buffers[0]

    fors = _collect(mod, lambda n: isinstance(n, tvm.tirx.For))
    kept = [f for f in fors if _as_int(f.extent) == M]
    chunk = [f for f in fors if _as_int(f.extent) == 2]
    assert len(kept) == 1 and len(chunk) == 1, f"expected one kept loop and one chunk loop, got {fors}"

    def _inside(outer, inner) -> bool:
        seen = []

        def visit(node):
            if node.same_as(inner):
                seen.append(node)

        tvm.tirx.stmt_functor.post_order_visit(outer, visit)
        return bool(seen)

    assert _inside(kept[0], chunk[0]), "the chunk loop must be nested inside the kept loop"
    init_stores = [
        store for store in _remaining_stores(mod, remaining) if isinstance(store.value, tvm.tirx.IntImm) and int(store.value.value) == N
    ]
    assert init_stores, "remaining must be initialized to the selected extent N"
    assert _inside(kept[0], init_stores[0]), "the 2D reset must live inside the kept loop"


@pytest.mark.pto
def test_real_stride_addresses_survive_vectorization():
    M, N, lanes = 4, 150, 128
    P = _ceil_to_lanes(N, lanes)
    mod, _ = _lower_capture(_k2d_row(M, N, lanes))

    calls = _vmi_calls(mod)
    assert "tl.vmi.vload" in calls and "tl.vmi.vstore" in calls

    strides = []
    for call in calls["tl.vmi.vload"] + calls["tl.vmi.vstore"]:
        for arg in call.args:
            if isinstance(arg, tvm.tirx.PrimExpr) and _subexpr_contains_constant(arg, P):
                strides.append(call)
                break
    assert strides, f"the padded row stride {P} must appear in the vector addresses"

    buffers = _buffers(mod)
    for name in ("a", "b"):
        assert name in buffers, f"buffer {name} must survive"
        shape = [_as_int(dim) for dim in buffers[name].shape]
        assert shape == [M * P], f"buffer {name} shape must stay as allocated, got {shape}"


@pytest.mark.pto
def test_explicit_padding_is_rejected_at_this_stage():
    E, lanes = 150, 128
    _, error = _lower_capture(_k_explicit_padding(E, lanes), expect_error=True)
    assert error is not None, "an if-guarded T.Parallel must still be rejected at this stage"
    message = str(error)
    assert "VerifyParallelToPTO" in message or "conditional" in message or "unsupported" in message, (
        f"expected the guard rejection diagnostic, got: {message}"
    )


# ---------------------------------------------------------------------------
# Sub/div/unary emission and lane-uniform folding
# ---------------------------------------------------------------------------


def _sub_kernel(n, lanes):
    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            b_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    c_ub[i] = a_ub[i] - b_ub[i]
            T.copy(c_ub, C)

    return main


@pytest.mark.pto
def test_vectorize_emits_vsub():
    source = _lower(_sub_kernel(256, 128), target="pto").kernel_source
    assert "pto.vmi.vsub(" in source


def _int_uniform_fold_kernel(n, lanes):
    @T.prim_func
    def main(A: T.Tensor((n,), "int32"), S: T.Tensor((1,), "int32"), C: T.Tensor((n,), "int32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "int32")
            s_ub = T.alloc_shared((1,), "int32")
            c_ub = T.alloc_shared((n,), "int32")
            T.copy(A, a_ub)
            T.copy(S, s_ub)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    # The scalar bind pins the uniform Sub into its own
                    # statement (an inline `a - (s-1)` is algebraically
                    # restructured into `(a+1) - s` by the analyzer). The
                    # bound value must stay a scalar Sub feeding the
                    # broadcast, not become a vsub of its own.
                    d: T.int32 = s_ub[0] - 1
                    c_ub[i] = a_ub[i] - d
            T.copy(c_ub, C)

    return main


@pytest.mark.pto
def test_vectorize_int_uniform_sub_folds_scalar():

    from tilelang.ascend import transform as atx

    # Run Vectorize directly on the captured post-Verify module and check
    # the emitted TIR: the inner uniform Sub must remain a scalar SubNode
    # inside the vbrc operand, not become a tl.vmi.vsub call of its own.
    with _verify_boundary_capture() as capture:
        _lower(_int_uniform_fold_kernel(256, 128), target="pto")
    mod = capture["module"]
    out = atx.VectorizeParallelToPTO()(mod)
    script = out["main"].script()
    assert script.count("tl.vmi.vsub") == 1, "expected exactly one vsub (the varying outer sub)\n" + script
    scalar_sub = re.search(r"d: T\.int32 = .* - 1", script)
    assert scalar_sub is not None, "expected the runtime-uniform sub to stay a scalar bind\n" + script


def _runtime_uniform_float_kernel(n, lanes):
    @T.prim_func
    def main(S: T.Tensor((2,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            s_ub = T.alloc_shared((2,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(S, s_ub)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    # Both operands are runtime lane-uniform floats: this
                    # stays scalar and must be rejected at Verify.
                    c_ub[i] = s_ub[0] / s_ub[1]
            T.copy(c_ub, C)

    return main


@pytest.mark.pto
def test_verify_runtime_uniform_float_div_rejected():
    with pytest.raises(
        Exception,
        match=r"runtime lane-uniform float computation.*scalar float ALU",
    ):
        _lower(_runtime_uniform_float_kernel(256, 128), target="pto")


def _unary_kernel(n, lanes, kind):
    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    if kind == 0:
                        c_ub[i] = T.abs(a_ub[i] - 0.5)
                    else:
                        c_ub[i] = T.sqrt(T.abs(a_ub[i]) + 0.25)
            T.copy(c_ub, C)

    return main


@pytest.mark.pto
def test_vectorize_emits_vabs():
    source = _lower(_unary_kernel(256, 128, 0), target="pto").kernel_source
    assert "pto.vmi.vabs(" in source


@pytest.mark.pto
def test_vectorize_emits_vsqrt():
    source = _lower(_unary_kernel(256, 128, 1), target="pto").kernel_source
    assert "pto.vmi.vsqrt(" in source


# ---------------------------------------------------------------------------
# Lane-use semantics for the uniform-float rejection
# ---------------------------------------------------------------------------


def _varying_bind_then_unary_kernel(n, lanes):
    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    # x is lane-varying through the bind; the uniform-float
                    # rejection must NOT fire for exp(x) even though the
                    # expression text does not mention the loop var.
                    x = a_ub[i] * 2.0
                    c_ub[i] = T.exp(x)
            T.copy(c_ub, C)

    return main


@pytest.mark.pto
def test_varying_bind_unary_accepted():
    source = _lower(_varying_bind_then_unary_kernel(256, 128), target="pto").kernel_source
    assert "pto.vmi.vexp(" in source
    assert "pto.vmi.vmul(" in source


def _chunk_uniform_float_kernel(n, lanes):
    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    # The address is uniform within a chunk (i // 128 * 128)
                    # but the expression mentions the loop var. The
                    # lane-use analysis classifies the load as uniform, so
                    # the float arithmetic must be rejected even though a
                    # syntactic loop-var check would have skipped it.
                    c_ub[i] = a_ub[(i // 128) * 128] + 1.0
            T.copy(c_ub, C)

    return main


@pytest.mark.pto
def test_chunk_uniform_float_rejected():
    with pytest.raises(
        Exception,
        match=r"runtime lane-uniform float computation.*scalar float ALU",
    ):
        _lower(_chunk_uniform_float_kernel(256, 128), target="pto")
