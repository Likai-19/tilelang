"""PTO slot storage capacity and exclusivity acceptance tests.

The reservation design's core acceptance surface (design §8.2, first
priority):

* one backing per fragment buffer, kept single across many sequential
  rounds (ping/pong shape) — never re-versioned per round;
* the reservation participates in the merged unified buffer: with several
  fragments the per-backing byte intervals [offset, offset + bytes) are
  pairwise disjoint and the merged total stays within the physical cap;
* both scheduling paths (auto-scheduled and manual) produce the same
  exclusivity;
* the backing's base is a real offset in the merged arena when other
  shared buffers precede it (non-zero base).

The byte intervals are recovered from the emitted
declare_fragment_buffer calls in the FINAL merged module: each
declaration's address operand is the backing's constant byte offset in
the merged arena (codegen prints constant offsets as plain integers).
Hardware independent: lowering only.
"""

from __future__ import annotations

import pytest

import tilelang.ascend.language as T
from tilelang.ascend import transform as ascend_transform
from tilelang.ascend.target import normalize_pto_target
from tilelang.engine.lower import lower as _lower
import tvm


# ---------------------------------------------------------------------------
# Capture helpers (same shape as the structure suites)
# ---------------------------------------------------------------------------


def _lower_capture(func, pass_name: str):
    """Lower with a capture wrapper around the named pass; the captured
    IRModule is the post-pass state (structure-suite pattern)."""
    from contextlib import contextmanager

    @contextmanager
    def _capture(pass_name: str):
        captured = {}
        original = getattr(ascend_transform, pass_name)

        def capturing_factory(*args, **kwargs):
            inner = original(*args, **kwargs)

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

    target = normalize_pto_target("pto")
    with _capture(pass_name) as captured:
        with target:
            _lower(func, target=target)
    return captured.get("module")


def _lower_capture_manual(func, pass_name: str):
    import tilelang

    from contextlib import contextmanager

    @contextmanager
    def _capture(pass_name: str):
        captured = {}
        original = getattr(ascend_transform, pass_name)

        def capturing_factory(*args, **kwargs):
            inner = original(*args, **kwargs)

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

    target = normalize_pto_target("pto")
    with _capture(pass_name) as captured:
        with target, tvm.transform.PassContext(
            config={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False}
        ):
            _lower(func, target=target)
    return captured.get("module")


def _iter_functions(mod):
    for func in mod.functions.values():
        yield func.body


def _declared_backing_ranges(mod):
    """(offset_bytes, size_bytes) per declare_fragment_buffer call. In
    the final merged module the address operand is
    tvm_access_ptr(dtype, arena, offset_elems, extent_elems, mask): the
    backing's arena placement is (offset, offset + count * lanes) in
    elements of the fragment dtype."""
    ranges = []

    def visit(node):
        if not isinstance(node, tvm.tirx.Call):
            return
        name = getattr(getattr(node, "op", None), "name", None)
        if name != "tl.vmi.declare_fragment_buffer":
            return
        offset_elems = 0
        elem_bits = 32
        ptr = node.args[0]
        if isinstance(ptr, tvm.tirx.Call):
            for arg in ptr.args:
                if isinstance(arg, tvm.tirx.IntImm) and arg.dtype.bits == 32:
                    pass
            # tvm_access_ptr(dtype_annot, arena_var, offset, extent, mask)
            ints = [a for a in ptr.args if isinstance(a, tvm.tirx.IntImm)]
            if len(ints) >= 2:
                offset_elems = ints[0].value
            text = str(ptr.args[0]) if ptr.args else ""
            if "float16" in text or "bfloat16" in text:
                elem_bits = 16
            if "int8" in text or "uint8" in text:
                elem_bits = 8
        count = getattr(node.args[1], "value", 0)
        lanes = getattr(node.args[2], "value", 0)
        elem_bytes = max(1, (elem_bits + 7) // 8)
        ranges.append((offset_elems * elem_bytes, count * lanes * elem_bytes))

    for body in _iter_functions(mod):
        tvm.tirx.stmt_functor.post_order_visit(body, visit)
    return ranges


def _merged_total(mod) -> int:
    """The merged shared.dyn arena size in bytes (the allocation's
    constant shape)."""
    best = 0

    def visit(node):
        nonlocal best
        if not isinstance(node, tvm.tirx.AllocBuffer):
            return
        if node.buffer.scope() != "shared.dyn":
            return
        total = 1
        ok = True
        for dim in node.buffer.shape:
            if isinstance(dim, tvm.tirx.IntImm):
                total *= dim.value
            else:
                ok = False
                break
        if ok:
            best = max(best, total * max(1, (node.buffer.dtype.bits + 7) // 8))

    for body in _iter_functions(mod):
        tvm.tirx.stmt_functor.post_order_visit(body, visit)
    return best


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def _ping_pong_rounds(rounds: int, lanes: int = 128):
    """One fragment kept across many sequential RMW rounds (ping/pong):
    the backing must stay a single allocation regardless of the round
    count."""

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
                for _r in range(rounds):
                    for i in T.Parallel(lanes):
                        keep[i] = keep[i] * T.float32(1.01)
                for i in T.Parallel(lanes):
                    b_ub[i] = keep[i]
            T.copy(b_ub, B)

    return main


def _two_fragments_with_preceding_ub(lanes: int = 128):
    """A large ordinary shared buffer plus two fragments: three separate
    reservations whose byte intervals must be pairwise disjoint in the
    merged arena, and the backings' bases must be non-zero (the ordinary
    buffer precedes them)."""

    @T.prim_func
    def main(
        A: T.Tensor((2 * lanes,), "float32"),
        B: T.Tensor((4 * lanes,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((2 * lanes,), "float32")
            b_ub = T.alloc_shared((4 * lanes,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                keep1 = T.alloc_fragment((lanes,), "float32")
                keep2 = T.alloc_fragment((2 * lanes,), "float32")
                for i in T.Parallel(lanes):
                    keep1[i] = a_ub[i]
                for i in T.Parallel(2 * lanes):
                    keep2[i] = a_ub[i % lanes]
                for i in T.Parallel(lanes):
                    b_ub[i] = keep1[i]
                for i in T.Parallel(2 * lanes):
                    b_ub[2 * lanes + i] = keep2[i]
            T.copy(b_ub, B)

    return main


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.pto
def test_backing_stays_single_across_rounds():
    """Ping/pong: the fragment survives many sequential rounds, but the
    final module declares its backing exactly once — the reservation is
    a single memory object, never re-versioned per round."""
    mod = _lower_capture(_ping_pong_rounds(rounds=6), "VerifyPTOCodegenLegality")
    declares = []
    for body in _iter_functions(mod):
        def visit(node):
            if isinstance(node, tvm.tirx.Call):
                name = getattr(getattr(node, "op", None), "name", None)
                if name == "tl.vmi.declare_fragment_buffer":
                    declares.append(node)
        tvm.tirx.stmt_functor.post_order_visit(body, visit)
    assert len(declares) == 1, (
        f"one declaration expected regardless of the round count, got {len(declares)}"
    )


@pytest.mark.pto
def test_backing_intervals_disjoint_auto_path():
    """Auto-scheduled: the per-backing byte intervals in the merged arena
    are pairwise disjoint (exclusivity has a direct address-level proof,
    not only the empty alias list)."""
    mod = _lower_capture(_two_fragments_with_preceding_ub(), "VerifyPTOCodegenLegality")
    ranges = _declared_backing_ranges(mod)
    assert len(ranges) >= 2, f"expected the two backings' declarations, got {ranges}"
    for a in range(len(ranges)):
        for b in range(a + 1, len(ranges)):
            (off_a, size_a), (off_b, size_b) = ranges[a], ranges[b]
            overlap = not (off_a + size_a <= off_b or off_b + size_b <= off_a)
            assert not overlap, (
                f"backing byte ranges overlap: {ranges[a]} vs {ranges[b]}"
            )


@pytest.mark.pto
def test_backing_intervals_disjoint_manual_path():
    """Manual scheduling path: same exclusivity statement."""
    mod = _lower_capture_manual(_two_fragments_with_preceding_ub(), "VerifyPTOCodegenLegality")
    ranges = _declared_backing_ranges(mod)
    assert len(ranges) >= 2, f"expected the two backings' declarations, got {ranges}"
    for a in range(len(ranges)):
        for b in range(a + 1, len(ranges)):
            (off_a, size_a), (off_b, size_b) = ranges[a], ranges[b]
            overlap = not (off_a + size_a <= off_b or off_b + size_b <= off_a)
            assert not overlap, (
                f"backing byte ranges overlap: {ranges[a]} vs {ranges[b]}"
            )


@pytest.mark.pto
def test_backing_base_nonzero_after_preceding_ub():
    """A large ordinary shared buffer precedes the fragments: the merged
    arena places the backings at non-zero byte offsets."""
    mod = _lower_capture(_two_fragments_with_preceding_ub(), "VerifyPTOCodegenLegality")
    ranges = _declared_backing_ranges(mod)
    assert len(ranges) >= 2
    assert all(off > 0 for off, _ in ranges), (
        f"backings expected after the ordinary buffer, got {ranges}"
    )


@pytest.mark.pto
def test_reservation_fits_merged_budget():
    """The merged total (reservation + ordinary buffers) stays within the
    physical unified buffer (216 KiB with a SIMT_VF context; the kernels
    here convert so the vecscope context applies)."""
    mod = _lower_capture(_two_fragments_with_preceding_ub(), "VerifyPTOCodegenLegality")
    total = _merged_total(mod)
    assert 0 < total <= 216 * 1024, (
        f"merged unified buffer total {total} exceeds the 216 KiB cap"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
