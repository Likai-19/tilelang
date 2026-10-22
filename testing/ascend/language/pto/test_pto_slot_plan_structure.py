"""PTO slot storage structure tests.

Hardware independent: each case lowers TileLang IR for the PTO target with
:func:`tilelang.engine.lower.lower` (TileLang IR only, no PTOAS/Bisheng device
compilation and no device execution) and inspects the modules produced by
``PlanPtoSlotStorage`` and ``VectorizeParallelToPTO`` through capture
wrappers.

Covered contract:

* PlanPtoSlotStorage declares one hidden ``shared.dyn`` backing per accessed
  ``local.fragment`` buffer and records a ``tl.pto_slot_plan`` annotation
  (entry count from the layout, vector lanes, aligned byte size, element
  dtype name, pre-created handle var, entry vector type) on the region's
  SBlock; the fragment allocation and its accesses are untouched at this
  stage;
* the entry count follows the layout (2D ``acc[M, VL]`` -> ``M`` entries;
  1D ``scores[E]`` -> ``ceil(E / VL)`` entries — never ``elements / VL``);
* Vectorize binds one ``declare_fragment_buffer(backing, count, lanes)``
  per plan inside the ``tl.simdvf_scope`` attr and converts fragment
  loads/stores into handle-based ``vload``/``vstore`` with
  ``element_offset = entry * lanes``, keeps two textual reads as two
  ``vload`` calls, and leaves no ``local.fragment`` access behind;
* a one-sided conditional store keeps the predicate: Vectorize merges it
  with the chunk mask (``mask_and``) instead of reading the old value and
  merging through ``vsel``;
* ``T.fill``/``T.copy`` touching a fragment buffer are expanded into
  ``T.Parallel`` loops before inference and convert through the same slot
  path (no ``tl.tileop.fill``/``tl.tileop.copy`` survives);
* region-level single-element fragment access, a lane-shifted fragment
  index (``B[i, j+1]``) and a fragment crossing two SIMD_VF regions are
  rejected by Verify;
* a slot reservation exceeding the unified-buffer capacity is rejected by
  the AutoSchedule budget.

Review-fix regressions additionally covered at the bottom of this file:

* a 2D fragment with multiple chunks per row (N = 2*VL) — whole-entry
  granularity proven by divisibility, not by the exact ``x*L`` product
  shape;
* an overlapping fragment self-copy is rejected by the expand pass, a
  disjoint one converts through the slot path;
* a whole-entry-shifted second subscript group (``keep[i + VL]``) is
  admitted when the shifted entry stays in range and rejected when it
  leaves the fragment (dual-sided entry interval);
* round-trip elimination stays enabled per buffer next to fragment
  accesses;
* a ``[VL]`` fragment accumulator inside an in-unit ``T.serial`` shell is
  never promoted to a carry vreg (B2);
* fragment entries that depend on outer serial loop variables (``f[k, j]``)
  prove their interval with the outer ranges bound — in range passes, out
  of range is still rejected.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

import tilelang.ascend.language as T
from tilelang.ascend import transform as ascend_transform
from tilelang.ascend.target import normalize_pto_target
from tilelang.engine.lower import lower as _lower
import tvm


# ---------------------------------------------------------------------------
# Capture helpers (same style as the existing PTO structure suites)
# ---------------------------------------------------------------------------


@contextmanager
def _capture_pass(pass_name: str):
    captured = {}
    original = getattr(ascend_transform, pass_name)

    def capturing_factory():
        inner = original()

        def run(mod):
            result = inner(mod)
            captured["module"] = result
            return result

        return run

    setattr(ascend_transform, pass_name, capturing_factory)
    try:
        yield captured
    finally:
        setattr(ascend_transform, pass_name, original)


def _lower_capture(func, pass_name: str, *, expect_error=False):
    target = normalize_pto_target("pto")
    error = None
    with _capture_pass(pass_name) as captured:
        try:
            with target:
                _lower(func, target=target)
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            error = exc
    if error is not None and not expect_error:
        raise error
    return captured.get("module"), error


def _lower_capture_manual(func, pass_name: str, *, expect_error=False):
    """Same as :func:`_lower_capture` but with AutoSchedule disabled, so the
    manual path runs (InferBufferAliases + Merge see the un-scheduled
    allocations)."""
    import tilelang

    target = normalize_pto_target("pto")
    error = None
    with _capture_pass(pass_name) as captured:
        try:
            with target, tvm.transform.PassContext(
                config={
                    tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False
                }
            ):
                _lower(func, target=target)
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            error = exc
    if error is not None and not expect_error:
        raise error
    return captured.get("module"), error


def _body_of(mod):
    assert len(mod.functions) == 1, f"expected a single function, got {list(mod.functions)}"
    return next(iter(mod.functions.values())).body


def _visit(mod, fn):
    tvm.tirx.stmt_functor.post_order_visit(_body_of(mod), fn)


def _vmi_calls(mod, name: str):
    calls = []

    def visit(node):
        if isinstance(node, tvm.tirx.Call):
            op_name = getattr(getattr(node, "op", None), "name", None)
            if op_name == f"tl.vmi.{name}":
                calls.append(node)

    _visit(mod, visit)
    return calls


def _sblocks(mod):
    blocks = []

    def visit(node):
        if isinstance(node, tvm.tirx.SBlock):
            blocks.append(node)

    _visit(mod, visit)
    return blocks


def _simdvf_block(mod):
    for block in _sblocks(mod):
        if block.name_hint == "SIMD_VF":
            return block
    return None


def _slot_plans(mod):
    plans = []
    for block in _sblocks(mod):
        for key, value in block.annotations.items():
            if key == "tl.pto_slot_plan":
                for plan in value:
                    plans.append((block, plan))
    return plans


def _collect_stores(mod, scope: str):
    stores = []

    def visit(node):
        if isinstance(node, tvm.tirx.BufferStore) and node.buffer.scope() == scope:
            stores.append(node)

    _visit(mod, visit)
    return stores


def _collect_loads(mod, scope: str):
    loads = []

    def visit(node):
        if isinstance(node, tvm.tirx.BufferLoad) and node.buffer.scope() == scope:
            loads.append(node)

    _visit(mod, visit)
    return loads


def _slot_vmi_calls(mod, vmi_name: str, token: str):
    """VMI calls of *vmi_name* whose printed form mentions *token* (the
    fragment handle name for handle-based accesses)."""

    calls = []

    def visit(node):
        if isinstance(node, tvm.tirx.Call):
            op_name = getattr(getattr(node, "op", None), "name", None)
            if op_name == f"tl.vmi.{vmi_name}":
                text = str(node)
                if token in text:
                    calls.append(node)

    _visit(mod, visit)
    return calls


def _declares(mod):
    return _vmi_calls(mod, "declare_fragment_buffer")


def _mem_bars(mod):
    bars = []

    def visit(node):
        if isinstance(node, tvm.tirx.Call):
            op_name = getattr(getattr(node, "op", None), "name", None)
            if op_name == "tl.simd.mem_bar":
                bars.append(node)

    _visit(mod, visit)
    return bars


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def _k_sv3_keep(CM: int, lanes: int, K: int = 3, dtype="float32"):
    """GEMV partial-sum keep: acc[CM, VL] lives across a serial K loop.

    ``xk`` is inlined as a lane-uniform UB scalar load (a ``local.var``
    accumulator is not accepted by the current input contract)."""

    @T.prim_func
    def main(
        A: T.Tensor((K, CM * lanes), dtype),
        X: T.Tensor((K,), dtype),
        Out: T.Tensor((CM * lanes,), dtype),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((K, CM * lanes), dtype)
            x_ub = T.alloc_shared((K,), dtype)
            out_ub = T.alloc_shared((CM * lanes,), dtype)
            T.copy(A, a_ub)
            T.copy(X, x_ub)
            with T.SimdVF(lanes=lanes):
                acc = T.alloc_fragment((CM, lanes), dtype)
                for i, j in T.Parallel(CM, lanes):
                    acc[i, j] = T.float32(0.0)
                for k in T.serial(K):
                    for i, j in T.Parallel(CM, lanes):
                        acc[i, j] = acc[i, j] + T.Cast(dtype, a_ub[k, i * lanes + j]) * T.Cast(dtype, x_ub[k])
                for i, j in T.Parallel(CM, lanes):
                    out_ub[i * lanes + j] = acc[i, j]
            T.copy(out_ub, Out)

    return main


def _k_scores_1d(E: int, lanes: int, dtype="float32"):
    """cf1-shaped keep: a 1D fragment over E elements, read back in full."""

    @T.prim_func
    def main(
        S: T.Tensor((E,), dtype),
        Out: T.Tensor((E,), dtype),
    ):
        with T.Kernel(1):
            s_ub = T.alloc_shared((E,), dtype)
            out_ub = T.alloc_shared((E,), dtype)
            T.copy(S, s_ub)
            with T.SimdVF(lanes=lanes):
                scores = T.alloc_fragment((E,), dtype)
                for i in T.Parallel(E):
                    scores[i] = s_ub[i]
                for i in T.Parallel(E):
                    out_ub[i] = scores[i]
            T.copy(out_ub, Out)

    return main


# ---------------------------------------------------------------------------
# Plan structure
# ---------------------------------------------------------------------------


def _no_fragment_tileops(mod, fragment_name: str):
    """No fill/copy/ascend_copy call that touches the fragment survives —
    the fragment ones are expanded before inference. Kernel-scope copies
    between ordinary UB buffers legitimately stay as ascend_copy, so the
    assertion matches on the fragment's name, not on the op alone."""
    tile_ops = []

    def visit(node):
        if isinstance(node, tvm.tirx.Call):
            name = getattr(getattr(node, "op", None), "name", None)
            if name in ("tl.tileop.fill", "tl.tileop.copy", "tl.tileop.ascend_copy"):
                if fragment_name in str(node):
                    tile_ops.append(name)

    _visit(mod, visit)
    return tile_ops


def _fragment_declare_args(declare):
    """Printed forms of the declare_fragment_buffer scalar arguments —
    IntImm wrappers vary (``8`` vs ``T.int64(8)``), so normalize."""
    args = []
    for a in declare.args:
        text = str(a)
        for ty in ("int64", "int32"):
            wrapper = f"T.{ty}("
            if text.startswith(wrapper) and text.endswith(")"):
                text = text[len(wrapper):-1]
                break
        args.append(text)
    return args


@pytest.mark.pto
def test_plan_declares_backing_and_annotation():
    CM, lanes = 24, 128
    mod, _ = _lower_capture(_k_sv3_keep(CM, lanes), "PlanPtoSlotStorage")
    plans = _slot_plans(mod)
    assert len(plans) == 1, f"expected one slot plan, got {len(plans)}"
    block, plan = plans[0]
    assert block.name_hint == "SIMD_VF"
    assert plan.fragment.name == "acc"
    assert plan.count == CM
    assert plan.vector_lanes == lanes
    assert plan.elem_dtype_name == "float32"
    assert plan.vector_type_name == f"float32x{lanes}"
    assert plan.fragment_mem_var.name == "acc_fragment_mem"
    # 24 entries * 128 lanes * 4 bytes = 12288, already 32B aligned.
    assert plan.bytes == 12288
    backing = plan.backing
    assert backing.scope() == "shared.dyn"
    assert int(backing.shape[0]) == CM * lanes
    # The fragment allocation is untouched at this stage and the region still
    # carries the scalar accesses.
    simdvf = _simdvf_block(mod)
    alloc_names = [buf.name for buf in simdvf.alloc_buffers]
    assert "acc" in alloc_names
    assert backing.name in alloc_names
    assert _collect_stores(mod, "local.fragment"), "fragment stores must survive Plan"


@pytest.mark.pto
def test_plan_count_1d_is_ceil_division():
    E, lanes = 200, 128  # non-divisible: 2 entries, not ceil(200*?/128)
    mod, _ = _lower_capture(_k_scores_1d(E, lanes), "PlanPtoSlotStorage")
    plans = _slot_plans(mod)
    assert len(plans) == 1
    plan = plans[0][1]
    assert plan.count == (E + lanes - 1) // lanes
    assert plan.bytes == 2 * lanes * 4


# ---------------------------------------------------------------------------
# Vectorize structure
# ---------------------------------------------------------------------------


@pytest.mark.pto
def test_vectorize_slot_roundtrip_2d():
    CM, lanes, K = 8, 128, 3
    mod, _ = _lower_capture(_k_sv3_keep(CM, lanes, K), "VectorizeParallelToPTO")

    # One declaration per plan, inside the vecscope, with the plan's
    # count/lanes and a whole-backing address operand.
    declares = _declares(mod)
    assert len(declares) == 1, [str(d) for d in declares]
    declare = declares[0]
    assert "acc_slot_backing" in str(declare)
    # The declaration carries the plan's exact count and lanes as its own
    # arguments (CM=8 would also appear as a substring of "128" in the
    # printed form, so match whole normalized argument strings).
    arg_texts = _fragment_declare_args(declare)
    assert str(CM) in arg_texts, f"declare args {arg_texts} lack the count {CM}"
    assert str(lanes) in arg_texts, f"declare args {arg_texts} lack the lanes {lanes}"

    handle = "acc_fragment_mem"
    stores = _slot_vmi_calls(mod, "vstore", handle)
    loads = _slot_vmi_calls(mod, "vload", handle)
    # Structural statement counts (TIR keeps loops, no unrolling): the
    # zero-init round contributes one store (broadcast value), the
    # accumulation round one load + one store inside its k/i loops, and
    # the write-back round one load.
    assert len(stores) == 2, [str(s) for s in stores]
    assert len(loads) == 2, [str(l) for l in loads]
    # Entry q lives at element offset q * lanes relative to the handle.
    for call in loads + stores:
        assert f"i * {lanes}" in str(call), str(call)

    # A vecscope mem_bar orders every backing-writing unit before the next
    # unit's backing reads (two writing units here: zero-init and the
    # accumulation loop).
    bars = _mem_bars(mod)
    assert len(bars) == 2, [str(b) for b in bars]
    assert all(str(b) == 'T.tl.simd.mem_bar("VST_VLD")' for b in bars)

    # No fragment access survives conversion; the fragment allocation is
    # consumed while the backing stays.
    assert not _collect_stores(mod, "local.fragment")
    assert not _collect_loads(mod, "local.fragment")
    simdvf = _simdvf_block(mod)
    alloc_names = [buf.name for buf in simdvf.alloc_buffers]
    assert "acc" not in alloc_names, "consumed fragment allocation must be removed"
    assert any(n.endswith("_slot_backing") for n in alloc_names)


@pytest.mark.pto
def test_vectorize_two_reads_stay_two_loads():
    """Two textual reads of the same fragment element are two backing
    vloads (no cross-access reuse at the TileLang level).

    The consumer is a read-modify-write unit (`keep[i] = keep[i] * keep[i]`)
    followed by a later read: the pure store->load round-trip elimination in
    the base pipeline legally forwards a single-store/single-load shape, so
    the RMW form is what pins down the no-reuse contract."""
    lanes = 128
    E = lanes

    @T.prim_func
    def main(A: T.Tensor((E,), "float32"), B: T.Tensor((E,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((E,), "float32")
            b_ub = T.alloc_shared((E,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                keep = T.alloc_fragment((E,), "float32")
                for i in T.Parallel(E):
                    keep[i] = a_ub[i]
                for i in T.Parallel(E):
                    keep[i] = keep[i] * keep[i]
                for i in T.Parallel(E):
                    b_ub[i] = keep[i]
            T.copy(b_ub, B)

    mod, _ = _lower_capture(main, "VectorizeParallelToPTO")
    # RMW round reads keep twice (the two textual reads), the final round
    # reads once: 3 handle vloads; two writing rounds: 2 handle vstores.
    handle = "keep_fragment_mem"
    loads = _slot_vmi_calls(mod, "vload", handle)
    stores = _slot_vmi_calls(mod, "vstore", handle)
    assert len(loads) == 3, [str(x) for x in loads]
    assert len(stores) == 2, [str(x) for x in stores]
    # The two RMW reads must be two distinct SSA values (no cross-access
    # reuse): the vmul consumes two different vload bindings.
    muls = _vmi_calls(mod, "vmul")
    assert len(muls) == 1
    mul = muls[0]
    lhs, rhs = str(mul.args[0]), str(mul.args[1])
    assert lhs != rhs, f"vmul must consume two distinct loads: {mul}"
    assert lhs.startswith("vload") and rhs.startswith("vload"), mul


# ---------------------------------------------------------------------------
# Verify rejections
# ---------------------------------------------------------------------------


@pytest.mark.pto
def test_region_level_fragment_access_rejected():
    lanes = 128
    E = lanes

    @T.prim_func
    def main(A: T.Tensor((E,), "float32"), B: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((E,), "float32")
            b_ub = T.alloc_shared((1,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                keep = T.alloc_fragment((E,), "float32")
                for i in T.Parallel(E):
                    keep[i] = a_ub[i]
                b_ub[0] = keep[0]
            T.copy(b_ub, B)

    _, error = _lower_capture(main, "VectorizeParallelToPTO", expect_error=True)
    assert error is not None
    text = str(error)
    # The layout pre-scan rejects region-level fragment access ahead of
    # Verify (A2). The bare statement is wrapped by the frontend into an
    # implicit extent-1 unit, so it lands either on the out-of-unit rule
    # or on the whole-entry dependency conflict — both name the buffer
    # and reject the single-element extract.
    assert ("accessed outside a converting T.Parallel body" in text or
            "cannot be served by whole-entry addressing" in text), text


@pytest.mark.pto
def test_fragment_fill_expands_to_slot_store():
    """T.fill on a fragment expands into a T.Parallel unit before inference
    and converts through the slot path: no tl.tileop.fill survives, the
    fill's value reaches the fragment store, and the fill-writing unit is
    ordered by a mem_bar. The consumer is read-modify-write so the base
    store->load round-trip elimination cannot forward the fill away (a
    pure store->single-load shape is legally forwarded without the slot).
    """
    lanes = 128
    E = lanes

    @T.prim_func
    def main(A: T.Tensor((E,), "float32"), B: T.Tensor((E,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((E,), "float32")
            b_ub = T.alloc_shared((E,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                keep = T.alloc_fragment((E,), "float32")
                T.fill(keep, T.float32(0.0))
                for i in T.Parallel(E):
                    keep[i] = keep[i] + a_ub[i]
                for i in T.Parallel(E):
                    b_ub[i] = keep[i]
            T.copy(b_ub, B)

    mod, _ = _lower_capture(main, "VectorizeParallelToPTO")
    tile_ops = _no_fragment_tileops(mod, "keep")
    assert not tile_ops, f"fragment fill/copy must not survive: {tile_ops}"
    assert not _collect_stores(mod, "local.fragment")
    assert not _collect_loads(mod, "local.fragment")
    assert len(_declares(mod)) == 1
    handle = "keep_fragment_mem"
    stores = _slot_vmi_calls(mod, "vstore", handle)
    loads = _slot_vmi_calls(mod, "vload", handle)
    # fill round: 1 store (broadcast value); RMW round: 1 load + 1 store;
    # read-out round: 1 load.
    assert len(stores) == 2, [str(s) for s in stores]
    assert len(loads) == 2, [str(l) for l in loads]
    # Two units write the backing (the fill unit and the RMW unit), so two
    # ordering mem_bars.
    bars = _mem_bars(mod)
    assert len(bars) == 2, [str(b) for b in bars]


@pytest.mark.pto
def test_fragment_copy_expands_to_slot_roundtrip():
    """T.copy out of a fragment expands into a T.Parallel unit and converts
    through the slot path: a handle load feeds an ordinary UB vstore. The
    producer is read-modify-write so the base store->load round-trip
    elimination cannot forward the fragment away before Vectorize."""
    lanes = 128
    E = lanes

    @T.prim_func
    def main(A: T.Tensor((E,), "float32"), B: T.Tensor((E,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((E,), "float32")
            b_ub = T.alloc_shared((E,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                keep = T.alloc_fragment((E,), "float32")
                for i in T.Parallel(E):
                    keep[i] = a_ub[i]
                for i in T.Parallel(E):
                    keep[i] = keep[i] * keep[i]
                T.copy(keep, b_ub)
            T.copy(b_ub, B)

    mod, _ = _lower_capture(main, "VectorizeParallelToPTO")
    handle = "keep_fragment_mem"
    loads = _slot_vmi_calls(mod, "vload", handle)
    stores = _slot_vmi_calls(mod, "vstore", handle)
    # producer round: 1 store; RMW round: 2 loads + 1 store; expanded copy
    # round: 1 load (feeding the UB side through the continuous path).
    assert len(stores) == 2, [str(s) for s in stores]
    assert len(loads) == 3, [str(s) for s in loads]
    assert not _collect_loads(mod, "local.fragment")


@pytest.mark.pto
def test_lane_shifted_fragment_index_rejected():
    lanes = 128
    M, N = 4, lanes

    @T.prim_func
    def main(A: T.Tensor((M * N,), "float32"), B: T.Tensor((M * N,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((M * N,), "float32")
            b_ub = T.alloc_shared((M * N,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                keep = T.alloc_fragment((M, N), "float32")
                for i, j in T.Parallel(M, N):
                    keep[i, j] = a_ub[i * N + j]
                for i, j in T.Parallel(M, N):
                    # j+1 shifts the element's storage lane by one — the
                    # whole-entry slot form cannot express it.
                    b_ub[i * N + j] = keep[i, j + 1]
            T.copy(b_ub, B)

    _, error = _lower_capture(main, "VectorizeParallelToPTO", expect_error=True)
    assert error is not None
    text = str(error)
    assert "storage lane" in text or "slot entry index" in text, text


@pytest.mark.pto
def test_fragment_crossing_two_regions_rejected():
    lanes = 128
    E = lanes

    @T.prim_func
    def main(A: T.Tensor((E,), "float32"), B: T.Tensor((E,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((E,), "float32")
            b_ub = T.alloc_shared((E,), "float32")
            T.copy(A, a_ub)
            # Allocated at kernel scope so both regions see it lexically;
            # the plan is recorded on the first region's SBlock only, so the
            # second region must reject the cross-region use.
            keep = T.alloc_fragment((E,), "float32")
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    keep[i] = a_ub[i]
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    b_ub[i] = keep[i]
            T.copy(b_ub, B)

    _, error = _lower_capture(main, "VectorizeParallelToPTO", expect_error=True)
    assert error is not None
    text = str(error)
    # The layout pre-scan rejects cross-region fragment use up front with
    # the ownership message (the old Verify message fired after planning).
    assert "more than one SIMD_VF region" in text or (
        "no slot plan in scope" in text and "shared across SIMD_VF" in text
    ), text


@pytest.mark.pto
def test_conditional_fragment_store_merges_mask():
    """cf1 shape: a single-sided conditional fragment store. Legalize
    lowers it to a predicated store; Vectorize merges the predicate mask and
    writes once through a mask_and(chunk, condition) vstore — no read of
    the old value, no vsel (inactive lanes keep the previous contents via
    the masked store)."""
    lanes = 128
    E = lanes

    @T.prim_func
    def main(A: T.Tensor((E,), "float32"), B: T.Tensor((E,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((E,), "float32")
            out_ub = T.alloc_shared((E,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                scores = T.alloc_fragment((E,), "float32")
                for i in T.Parallel(E):
                    scores[i] = a_ub[i]
                for i in T.Parallel(E):
                    if a_ub[i] > T.float32(0.5):
                        scores[i] = T.float32(-1.0)
                for i in T.Parallel(E):
                    out_ub[i] = scores[i]
            T.copy(out_ub, B)

    mod, _ = _lower_capture(main, "VectorizeParallelToPTO")
    handle = "scores_fragment_mem"
    loads = _slot_vmi_calls(mod, "vload", handle)
    stores = _slot_vmi_calls(mod, "vstore", handle)
    vsels = _vmi_calls(mod, "vsel")
    mask_ands = _vmi_calls(mod, "mask_and")
    # init round: 1 store; conditional round: 1 store with the merged mask
    # and no old-value load; write-back round: 1 load.
    assert len(stores) == 2, [str(s) for s in stores]
    assert len(loads) == 1, [str(s) for s in loads]
    assert len(vsels) == 0, "the write-back must not merge through vsel"
    assert len(mask_ands) == 1, "the predicate must merge into the store mask"
    # The conditional store consumes the merged mask (the EmitBind'd
    # mask_and result), not the bare chunk mask variable.
    cond_store = stores[1]
    assert "mand" in str(cond_store.args[3]), str(cond_store)


# ---------------------------------------------------------------------------
# Capacity budget
# ---------------------------------------------------------------------------


@pytest.mark.pto
def test_slot_reservation_over_capacity_rejected():
    lanes = 128
    # 1024 entries * 128 lanes * 4 B = 512 KiB > the 248 KiB unified buffer.
    # The kernel keeps every other UB allocation tiny so the slot deduction
    # is unambiguously the failing check.
    CM = 1024

    @T.prim_func
    def main(B: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            b_ub = T.alloc_shared((1,), "float32")
            with T.SimdVF(lanes=lanes):
                keep = T.alloc_fragment((CM, lanes), "float32")
                for i, j in T.Parallel(CM, lanes):
                    keep[i, j] = T.float32(1.0)
            b_ub[0] = T.float32(0.0)
            T.copy(b_ub, B)

    _, error = _lower_capture(main, "PlanPtoSlotStorage", expect_error=True)
    assert error is not None
    assert "PTO slot storage reserves" in str(error)


@pytest.mark.pto
def test_kernel_scope_fragment_reservation_budgeted():
    """A fragment buffer allocated at kernel scope (outside the SIMD_VF
    block) still gets its plan recorded on the region that uses it, so the
    AutoSchedule budget sees the reservation through the kernel-root plan
    scan and rejects it over capacity."""
    lanes = 128
    CM = 1024  # 512 KiB reservation, over the 248 KiB unified buffer

    @T.prim_func
    def main(B: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            keep = T.alloc_fragment((CM, lanes), "float32")
            b_ub = T.alloc_shared((1,), "float32")
            with T.SimdVF(lanes=lanes):
                for i, j in T.Parallel(CM, lanes):
                    keep[i, j] = T.float32(1.0)
            b_ub[0] = T.float32(0.0)
            T.copy(b_ub, B)

    _, error = _lower_capture(main, "PlanPtoSlotStorage", expect_error=True)
    assert error is not None
    assert "PTO slot storage reserves" in str(error)


@pytest.mark.pto
def test_manual_schedule_backing_stays_unaliased():
    """With AutoSchedule disabled, InferBufferAliases runs after Vectorize
    consumed the slot plans: the backing must stay address-exclusive
    through its declare_fragment_buffer operand, not through the already
    consumed plan annotation."""
    lanes = 128
    E = lanes

    @T.prim_func
    def main(A: T.Tensor((E,), "float32"), Out: T.Tensor((E,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((E,), "float32")
            b_ub = T.alloc_shared((E,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                keep = T.alloc_fragment((E,), "float32")
                for i in T.Parallel(E):
                    keep[i] = a_ub[i]
                for i in T.Parallel(E):
                    keep[i] = keep[i] + keep[i]
                for i in T.Parallel(E):
                    b_ub[i] = keep[i]
            T.copy(b_ub, Out)

    mod, _ = _lower_capture_manual(main, "InferBufferAliases")
    alias_maps = []

    def visit(node):
        if isinstance(node, tvm.tirx.AttrStmt) and node.attr_key == "tl.buffer_alias_map":
            alias_maps.append(node)

    # The manual path splits the module into a host wrapper plus the device
    # kernel; scan every function's body.
    for func in mod.functions.values():
        tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    assert alias_maps, "InferBufferAliases must produce a tl.buffer_alias_map"
    backing_entries = {}
    for attr in alias_maps:
        try:
            items = attr.node.items()
        except AttributeError:
            continue
        for var, aliases in items:
            if "_slot_backing" in str(var):
                backing_entries[str(var)] = [str(a) for a in aliases]
    assert backing_entries, "the slot backing must appear in the alias map"
    for name, aliases in backing_entries.items():
        assert aliases == [], f"backing {name} must stay address-exclusive, got {aliases}"


# ---------------------------------------------------------------------------
# Review-fix regressions (whole-entry granularity, overlap rejection,
# per-buffer round-trip gating, fragment shell carry, outer-loop entry
# interval, fragment multi-subscript collection)
# ---------------------------------------------------------------------------


@pytest.mark.pto
def test_2d_fragment_multiple_chunks_per_row():
    """A 2D fragment wider than one vector (N = 2*VL) vectorizes with two
    chunks per row: the backing offset simplifies to i*(Q*L) + q*L, and the
    declaration checker must prove whole-entry granularity by divisibility
    instead of matching the exact ``x*L`` product shape."""
    lanes = 128
    M, N = 4, 2 * lanes

    @T.prim_func
    def main(A: T.Tensor((M * N,), "float32"), B: T.Tensor((M * N,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((M * N,), "float32")
            b_ub = T.alloc_shared((M * N,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                keep = T.alloc_fragment((M, N), "float32")
                for i, j in T.Parallel(M, N):
                    keep[i, j] = a_ub[i * N + j]
                for i, j in T.Parallel(M, N):
                    b_ub[i * N + j] = keep[i, j] * T.float32(2.0)
            T.copy(b_ub, B)

    mod, _ = _lower_capture(main, "PlanPtoSlotStorage")
    plans = _slot_plans(mod)
    assert len(plans) == 1
    assert plans[0][1].count == M * 2, "N=2*VL must reserve two entries per row"

    mod, _ = _lower_capture(main, "VectorizeParallelToPTO")
    handle = "keep_fragment_mem"
    stores = _slot_vmi_calls(mod, "vstore", handle)
    loads = _slot_vmi_calls(mod, "vload", handle)
    assert len(stores) == 1, [str(s) for s in stores]
    assert len(loads) == 1, [str(l) for l in loads]
    # The distributed offset i*(Q*L) + q*L with Q=2, L=128.
    for call in loads + stores:
        text = str(call)
        assert f"i * {2 * lanes}" in text or f"q * {lanes}" in text, text
    assert not _collect_stores(mod, "local.fragment")
    assert not _collect_loads(mod, "local.fragment")


@pytest.mark.pto
def test_fragment_self_copy_overlap_rejected():
    """An overlapping fragment self-copy expands into parallel units whose
    chunk order is undefined — a data race. ExpandPtoFragmentFillCopy
    rejects it with a classified diagnostic before any conversion."""
    lanes = 128
    E = 4 * lanes

    @T.prim_func
    def main(A: T.Tensor((E,), "float32"), B: T.Tensor((E,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((E,), "float32")
            b_ub = T.alloc_shared((E,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                keep = T.alloc_fragment((E,), "float32")
                for i in T.Parallel(E):
                    keep[i] = a_ub[i]
                T.copy(keep[0 : 2 * lanes], keep[lanes : 3 * lanes])
                for i in T.Parallel(E):
                    b_ub[i] = keep[i]
            T.copy(b_ub, B)

    _, error = _lower_capture(main, "ExpandPtoFragmentFillCopy", expect_error=True)
    assert error is not None
    text = str(error)
    assert "overlapping self-copy" in text, text


@pytest.mark.pto
def test_fragment_disjoint_self_copy_ok():
    """A disjoint fragment self-copy is a legal entry shift: it expands, is
    collected as a second subscript group (no structural-equality crash),
    and converts through the slot path."""
    lanes = 128
    E = 4 * lanes

    @T.prim_func
    def main(A: T.Tensor((E,), "float32"), B: T.Tensor((E,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((E,), "float32")
            b_ub = T.alloc_shared((E,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                keep = T.alloc_fragment((E,), "float32")
                for i in T.Parallel(E):
                    keep[i] = a_ub[i]
                T.copy(keep[0:lanes], keep[2 * lanes : 3 * lanes])
                for i in T.Parallel(E):
                    b_ub[i] = keep[i]
            T.copy(b_ub, B)

    mod, _ = _lower_capture(main, "VectorizeParallelToPTO")
    tile_ops = _no_fragment_tileops(mod, "keep")
    assert not tile_ops, f"fragment fill/copy must not survive: {tile_ops}"
    handle = "keep_fragment_mem"
    # producer: 1 store; expanded self-copy: 1 load + 1 store (entry 2);
    # read-out: 1 load.
    assert len(_slot_vmi_calls(mod, "vstore", handle)) == 2
    assert len(_slot_vmi_calls(mod, "vload", handle)) == 2
    assert not _collect_stores(mod, "local.fragment")
    assert not _collect_loads(mod, "local.fragment")


@pytest.mark.pto
def test_fragment_entry_shift_add_ok():
    """A whole-entry-shifted second subscript group (keep[i + VL] with the
    shifted index still inside the fragment) is admitted: the entry is a
    plain +1 shift, the lane is unchanged, and the interval proves."""
    lanes = 128

    @T.prim_func
    def main(A: T.Tensor((2 * lanes,), "float32"), B: T.Tensor((2 * lanes,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((2 * lanes,), "float32")
            b_ub = T.alloc_shared((2 * lanes,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                keep = T.alloc_fragment((2 * lanes,), "float32")
                for i in T.Parallel(2 * lanes):
                    keep[i] = a_ub[i]
                for i in T.Parallel(lanes):
                    keep[i] = keep[i] + keep[i + lanes]
                for i in T.Parallel(2 * lanes):
                    b_ub[i] = keep[i]
            T.copy(b_ub, B)

    mod, _ = _lower_capture(main, "VectorizeParallelToPTO")
    handle = "keep_fragment_mem"
    # producer: 1 store; shifted add: 2 loads (entries 0 and 1) + 1 store;
    # read-out: 1 load.
    assert len(_slot_vmi_calls(mod, "vstore", handle)) == 2
    assert len(_slot_vmi_calls(mod, "vload", handle)) == 3
    assert not _collect_stores(mod, "local.fragment")


@pytest.mark.pto
def test_fragment_entry_shift_out_of_range_rejected():
    """The dual-sided entry interval: keep[i + VL] over the full domain
    reads past the fragment and must be rejected by Verify (the shifted
    entry reaches the entry count)."""
    lanes = 128

    @T.prim_func
    def main(A: T.Tensor((2 * lanes,), "float32"), B: T.Tensor((2 * lanes,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((2 * lanes,), "float32")
            b_ub = T.alloc_shared((2 * lanes,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                keep = T.alloc_fragment((2 * lanes,), "float32")
                for i in T.Parallel(2 * lanes):
                    keep[i] = a_ub[i]
                for i in T.Parallel(2 * lanes):
                    keep[i] = keep[i] + keep[i + lanes]
                for i in T.Parallel(2 * lanes):
                    b_ub[i] = keep[i]
            T.copy(b_ub, B)

    _, error = _lower_capture(main, "VectorizeParallelToPTO", expect_error=True)
    assert error is not None
    text = str(error)
    assert "slot entry index" in text and ("reach or exceed" in text or "negative" in text), text


@pytest.mark.pto
def test_ub_roundtrip_eliminated_alongside_fragment():
    """Round-trip elimination is gated per buffer, not per region: a UB
    temp round-trip living next to fragment accesses is still forwarded
    (the fragment's own accesses never enter the rules)."""
    lanes = 128

    @T.prim_func
    def main(A: T.Tensor((lanes,), "float32"), B: T.Tensor((lanes,), "float32"),
             C: T.Tensor((lanes,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((lanes,), "float32")
            b_ub = T.alloc_shared((lanes,), "float32")
            c_ub = T.alloc_shared((lanes,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                keep = T.alloc_fragment((lanes,), "float32")
                tmp = T.alloc_shared((lanes,), "float32")
                for i in T.Parallel(lanes):
                    keep[i] = a_ub[i]
                for i in T.Parallel(lanes):
                    tmp[i] = a_ub[i] * T.float32(2.0)
                for i in T.Parallel(lanes):
                    b_ub[i] = tmp[i]
                for i in T.Parallel(lanes):
                    c_ub[i] = keep[i]
            T.copy(b_ub, B)
            T.copy(c_ub, C)

    mod, _ = _lower_capture(main, "VectorizeParallelToPTO")

    def buffer_accesses(name):
        hits = []

        def visit(node):
            if isinstance(node, (tvm.tirx.BufferLoad, tvm.tirx.BufferStore)):
                if node.buffer.name == name:
                    hits.append(node)

        _visit(mod, visit)
        return hits

    # The temp's round trip is gone (store forwarded into the copy
    # consumer); the fragment converts through its handle.
    assert not buffer_accesses("tmp"), "the UB temp round trip must be eliminated"
    handle = "keep_fragment_mem"
    assert len(_slot_vmi_calls(mod, "vstore", handle)) == 1
    assert len(_slot_vmi_calls(mod, "vload", handle)) == 1
    # The forwarded value still reaches b_ub: one plain UB store survives.
    assert len(buffer_accesses("b_ub")) >= 1


@pytest.mark.pto
def test_fragment_shell_accumulator_converts_via_slot():
    """A [VL] fragment accumulator updated inside an in-unit T.serial loop
    is NOT a carry candidate (B2): its read-modify-write goes through the
    slot backing, the shell is preserved, and no carry vreg/writeback is
    produced for it. The shell's own iterations are ordered: every
    backing-writing iteration closes with its own VST_VLD bar (the
    unit-level bar fires only after the whole unit)."""
    lanes = 128
    K = 5

    @T.prim_func
    def main(W: T.Tensor((K,), "float32"), B: T.Tensor((lanes,), "float32")):
        with T.Kernel(1):
            w_ub = T.alloc_shared((K,), "float32")
            b_ub = T.alloc_shared((lanes,), "float32")
            T.copy(W, w_ub)
            with T.SimdVF(lanes=lanes):
                acc = T.alloc_fragment((lanes,), "float32")
                for i in T.Parallel(lanes):
                    acc[i] = T.float32(0.0)
                for i in T.Parallel(lanes):
                    for k in T.serial(K):
                        acc[i] = acc[i] + w_ub[k]
                for i in T.Parallel(lanes):
                    b_ub[i] = acc[i]
            T.copy(b_ub, B)

    mod, _ = _lower_capture(main, "VectorizeParallelToPTO")
    assert "acc_carry" not in str(_body_of(mod)), "a fragment must never become a carry vreg"
    handle = "acc_fragment_mem"
    # init: 1 store; in-shell RMW: 1 load + 1 store; read-out: 1 load.
    assert len(_slot_vmi_calls(mod, "vstore", handle)) == 2
    assert len(_slot_vmi_calls(mod, "vload", handle)) == 2
    assert not _collect_stores(mod, "local.fragment")
    assert not _collect_loads(mod, "local.fragment")
    # Two unit-level bars (init unit, shell unit) plus one bar inside the
    # shell body — the per-iteration ordering point.
    bars = _mem_bars(mod)
    assert len(bars) == 3, [str(b) for b in bars]
    bars_inside_for = []

    def check_for(node):
        if isinstance(node, tvm.tirx.For):
            tvm.tirx.stmt_functor.post_order_visit(
                node.body,
                lambda n: bars_inside_for.append(n)
                if isinstance(n, tvm.tirx.Call)
                and getattr(getattr(n, "op", None), "name", "")
                == "tl.simd.mem_bar"
                else None,
            )

    _visit(mod, check_for)
    assert bars_inside_for, (
        "a backing-writing shell must close each iteration with a mem_bar"
    )


@pytest.mark.pto
def test_outer_serial_fragment_entry_ok():
    """The entry interval proof binds outer serial loop ranges: f[k, j]
    with k a region-level serial var and the fragment holding K entries
    lowers through both Verify and Vectorize (no 'can reach or exceed the
    entry count' rejection)."""
    lanes = 128
    K = 4

    @T.prim_func
    def main(A: T.Tensor((K,), "float32"), B: T.Tensor((K * lanes,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((K,), "float32")
            b_ub = T.alloc_shared((K * lanes,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                keep = T.alloc_fragment((K, lanes), "float32")
                for k in T.serial(K):
                    for j in T.Parallel(lanes):
                        keep[k, j] = a_ub[k]
                for k in T.serial(K):
                    for j in T.Parallel(lanes):
                        b_ub[k * lanes + j] = keep[k, j]
            T.copy(b_ub, B)

    mod, _ = _lower_capture(main, "VectorizeParallelToPTO")
    handle = "keep_fragment_mem"
    assert len(_slot_vmi_calls(mod, "vstore", handle)) == 1
    assert len(_slot_vmi_calls(mod, "vload", handle)) == 1
    assert not _collect_stores(mod, "local.fragment")
    assert not _collect_loads(mod, "local.fragment")


@pytest.mark.pto
def test_outer_serial_fragment_entry_out_of_range_rejected():
    """The interval bound stays real with outer ranges bound: f[k, j] with
    k covering more rows than the fragment has entries is rejected."""
    lanes = 128
    K = 4

    @T.prim_func
    def main(A: T.Tensor((K,), "float32"), B: T.Tensor((K * lanes,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((K,), "float32")
            b_ub = T.alloc_shared((K * lanes,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                keep = T.alloc_fragment((2, lanes), "float32")
                for k in T.serial(K):
                    for j in T.Parallel(lanes):
                        keep[k, j] = a_ub[k]
                for k in T.serial(K):
                    for j in T.Parallel(lanes):
                        b_ub[k * lanes + j] = keep[k, j]
            T.copy(b_ub, B)

    _, error = _lower_capture(main, "VectorizeParallelToPTO", expect_error=True)
    assert error is not None
    text = str(error)
    assert "slot entry index" in text and "reach or exceed" in text, text


@pytest.mark.pto
def test_split_region_fragment_entry_ok():
    """The vecscope split hoists a UB-carrying serial loop out of the
    region; the parts converted inside it still see the hoisted loop's
    range, so a fragment entry depending on it (keep[t, i]) proves its
    interval in both Verify (original tree) and Vectorize (split tree)."""
    lanes = 128
    TK = 4

    @T.prim_func
    def main(A: T.Tensor((TK * lanes,), "float32"), B: T.Tensor((lanes,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((TK * lanes,), "float32")
            y_ub = T.alloc_shared((lanes,), "float32")
            b_ub = T.alloc_shared((lanes,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                keep = T.alloc_fragment((TK, lanes), "float32")
                for t in T.serial(TK):
                    for i in T.Parallel(lanes):
                        keep[t, i] = a_ub[t * lanes + i]
                for i in T.Parallel(lanes):
                    y_ub[i] = T.float32(0.0)
                for t in T.serial(TK):
                    for i in T.Parallel(lanes):
                        y_ub[i] = y_ub[i] + keep[t, i]
                for i in T.Parallel(lanes):
                    b_ub[i] = y_ub[i]
            T.copy(b_ub, B)

    mod, _ = _lower_capture(main, "VectorizeParallelToPTO")
    handle = "keep_fragment_mem"
    assert len(_slot_vmi_calls(mod, "vstore", handle)) == 1
    assert len(_slot_vmi_calls(mod, "vload", handle)) == 1
    assert not _collect_stores(mod, "local.fragment")
    assert not _collect_loads(mod, "local.fragment")
    # The split produced more than one vecscope: each part declares its own
    # handle over the shared backing.
    assert len(_declares(mod)) >= 2, [str(d) for d in _declares(mod)]


# ---------------------------------------------------------------------------
# Joint layout solver (R4): B1 lane-on-first-dim, B5 floating, A2
# out-of-region access, proven conflicts, kernel-level entry ranges,
# cross-unit WAR bars, unaligned fill/copy expansion
# ---------------------------------------------------------------------------


@pytest.mark.pto
def test_solver_lane_dim_on_first_logical_dim():
    """B1: orientation-consistent access with lanes on the FIRST logical
    dimension (in_ub[j, i] forces the unit to vectorize along i). The
    joint solver picks lane dim 0 — 8 entries for f[256, 4] at VL=128
    (the byte-minimal choice; lane dim 1 would need 256)."""
    lanes = 128

    @T.prim_func
    def main(A: T.Tensor((4, 256), "float32"), B: T.Tensor((4, 256), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((4, 256), "float32")
            b_ub = T.alloc_shared((4, 256), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                keep = T.alloc_fragment((256, 4), "float32")
                for i, j in T.Parallel(256, 4):
                    keep[i, j] = a_ub[j, i]
                for i, j in T.Parallel(256, 4):
                    b_ub[j, i] = keep[i, j] * T.float32(2.0)
            T.copy(b_ub, B)

    mod, _ = _lower_capture(main, "PlanPtoSlotStorage")
    plans = _slot_plans(mod)
    assert len(plans) == 1
    assert plans[0][1].count == 8, (
        f"lane dim 0 reserves ceil(256/128)*4 = 8 entries, got {plans[0][1].count}"
    )
    mod, _ = _lower_capture(main, "VectorizeParallelToPTO")
    assert not _collect_stores(mod, "local.fragment")
    assert not _collect_loads(mod, "local.fragment")
    handle = "keep_fragment_mem"
    assert len(_slot_vmi_calls(mod, "vstore", handle)) == 1
    assert len(_slot_vmi_calls(mod, "vload", handle)) == 1


@pytest.mark.pto
def test_solver_floating_fragment_gets_layout():
    """B5: an allocated-but-unaccessed fragment (floating) gets a solved
    layout ahead of the engine's step 0, so the FullyReplicated seed and
    the merge-point FATAL are unreachable."""
    lanes = 128

    @T.prim_func
    def main(A: T.Tensor((lanes,), "float32"), B: T.Tensor((lanes,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((lanes,), "float32")
            b_ub = T.alloc_shared((lanes,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                keep = T.alloc_fragment((lanes,), "float32")
                unused = T.alloc_fragment((4, 128), "float32")
                for i in T.Parallel(lanes):
                    keep[i] = a_ub[i]
                for i in T.Parallel(lanes):
                    b_ub[i] = keep[i]
            T.copy(b_ub, B)

    mod, _ = _lower_capture(main, "VectorizeParallelToPTO")
    # Only the accessed fragment gets a plan/declaration; the floating one
    # just needs a layout to exist (no FATAL during inference).
    assert len(_declares(mod)) == 1


@pytest.mark.pto
def test_solver_kernel_scope_fragment_access_rejected():
    """A2: a fragment used inside a converting region and also at kernel
    scope (outside any region) is rejected by the pre-scan with the
    ownership message."""
    lanes = 128

    @T.prim_func
    def main(A: T.Tensor((lanes,), "float32"), B: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((lanes,), "float32")
            b_ub = T.alloc_shared((1,), "float32")
            T.copy(A, a_ub)
            keep = T.alloc_fragment((lanes,), "float32")
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(lanes):
                    keep[i] = a_ub[i]
            b_ub[0] = keep[0]
            T.copy(b_ub, B)

    _, error = _lower_capture(main, "VectorizeParallelToPTO", expect_error=True)
    assert error is not None
    text = str(error)
    assert "accessed outside a converting T.Parallel body" in text, text


@pytest.mark.pto
def test_solver_proven_conflict_diagnosed():
    """Two units sharing one fragment but forced onto opposite
    vectorization directions by their UB accesses: a proven parity
    conflict with the originating unit in the message."""
    lanes = 128

    @T.prim_func
    def main(
        A: T.Tensor((128, 128), "float32"),
        C: T.Tensor((128, 128), "float32"),
        D: T.Tensor((128, 128), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((128, 128), "float32")
            c_ub = T.alloc_shared((128, 128), "float32")
            d_ub = T.alloc_shared((128, 128), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                keep = T.alloc_fragment((128, 128), "float32")
                # a[i, j] is continuous along j: forces the inner
                # direction, so keep's lane dim must be 1.
                for i, j in T.Parallel(128, 128):
                    keep[i, j] = a_ub[i, j]
                # c[j, i] is continuous along i only: forces the outer
                # direction, so keep's lane dim must be 0 — a proven
                # conflict with the first unit.
                for i, j in T.Parallel(128, 128):
                    c_ub[j, i] = keep[i, j]
                for i, j in T.Parallel(128, 128):
                    d_ub[i, j] = T.float32(0.0)
            T.copy(c_ub, C)
            T.copy(d_ub, D)

    _, error = _lower_capture(main, "VectorizeParallelToPTO", expect_error=True)
    assert error is not None
    text = str(error)
    assert "proven layout conflict" in text, text


@pytest.mark.pto
def test_kernel_level_loop_fragment_entry_ok():
    """A4 (extended): a fragment entry depending on a serial loop at
    KERNEL scope (outside the region) proves its interval on both sides —
    previously rejected because neither side bound the range."""
    lanes = 128
    K = 4

    @T.prim_func
    def main(A: T.Tensor((K * lanes,), "float32"), B: T.Tensor((K * lanes,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((K * lanes,), "float32")
            b_ub = T.alloc_shared((K * lanes,), "float32")
            T.copy(A, a_ub)
            keep = T.alloc_fragment((K, lanes), "float32")
            with T.SimdVF(lanes=lanes):
                for b in T.serial(K):
                    for j in T.Parallel(lanes):
                        keep[b, j] = a_ub[b * lanes + j]
                for b in T.serial(K):
                    for j in T.Parallel(lanes):
                        b_ub[b * lanes + j] = keep[b, j]
            T.copy(b_ub, B)

    mod, _ = _lower_capture(main, "VectorizeParallelToPTO")
    handle = "keep_fragment_mem"
    assert len(_slot_vmi_calls(mod, "vstore", handle)) == 1
    assert len(_slot_vmi_calls(mod, "vload", handle)) == 1
    assert not _collect_stores(mod, "local.fragment")


@pytest.mark.pto
def test_cross_unit_war_bar():
    """A1: a unit that only reads the backing followed by a unit that
    writes it gets a WAR (VLD_VST) bar before the writer, in addition to
    the writer's trailing VST_VLD."""
    lanes = 128

    @T.prim_func
    def main(A: T.Tensor((lanes,), "float32"), B: T.Tensor((lanes,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((lanes,), "float32")
            b_ub = T.alloc_shared((lanes,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                keep = T.alloc_fragment((lanes,), "float32")
                for i in T.Parallel(lanes):
                    keep[i] = a_ub[i]
                for i in T.Parallel(lanes):
                    b_ub[i] = keep[i] + T.float32(1.0)
                for i in T.Parallel(lanes):
                    keep[i] = b_ub[i] * T.float32(2.0)
                for i in T.Parallel(lanes):
                    b_ub[i] = keep[i]
            T.copy(b_ub, B)

    mod, _ = _lower_capture(main, "VectorizeParallelToPTO")
    bars = [str(b) for b in _mem_bars(mod)]
    assert any("VLD_VST" in b for b in bars), bars
    # Unit sequence: write(init), read, write(needs WAR before + RAW
    # after), read. Expected: VST_VLD after init; VLD_VST before the
    # third; VST_VLD after the third.
    assert len([b for b in bars if "VLD_VST" in b]) == 1, bars
    assert len([b for b in bars if "VST_VLD" in b]) == 2, bars


@pytest.mark.pto
def test_unaligned_fill_expands_predicated():
    """G2: an unaligned fill region (min 64 at VL=128) expands to the
    predicated whole-entry form instead of being rejected."""
    lanes = 128

    @T.prim_func
    def main(B: T.Tensor((2 * lanes,), "float32")):
        with T.Kernel(1):
            b_ub = T.alloc_shared((2 * lanes,), "float32")
            with T.SimdVF(lanes=lanes):
                keep = T.alloc_fragment((2 * lanes,), "float32")
                T.fill(keep[lanes // 2 : lanes // 2 + lanes], T.float32(7.0))
                for i in T.Parallel(lanes):
                    keep[i] = keep[i] + T.float32(1.0)
                for i in T.Parallel(2 * lanes):
                    b_ub[i] = keep[i]
            T.copy(b_ub, B)

    mod, _ = _lower_capture(main, "VectorizeParallelToPTO")
    handle = "keep_fragment_mem"
    # predicated fill round: 1 store (mask guards the unaligned head);
    # RMW round: 1 load + 1 store; read-out: 1 load.
    assert len(_slot_vmi_calls(mod, "vstore", handle)) == 2
    assert len(_slot_vmi_calls(mod, "vload", handle)) == 2
    assert not _collect_stores(mod, "local.fragment")


@pytest.mark.pto
def test_unaligned_ub_to_fragment_copy_rejected():
    """G2: an unaligned fragment destination in a UB-to-fragment copy
    cannot stay whole-entry without reading past the source — rejected."""
    lanes = 128

    @T.prim_func
    def main(A: T.Tensor((lanes,), "float32"), B: T.Tensor((2 * lanes,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((lanes,), "float32")
            b_ub = T.alloc_shared((2 * lanes,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                keep = T.alloc_fragment((2 * lanes,), "float32")
                T.copy(a_ub[0:lanes], keep[lanes // 2 : lanes // 2 + lanes])
                for i in T.Parallel(2 * lanes):
                    b_ub[i] = keep[i]
            T.copy(b_ub, B)

    _, error = _lower_capture(main, "ExpandPtoFragmentFillCopy", expect_error=True)
    assert error is not None
    text = str(error)
    assert "unaligned fragment region" in text, text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
