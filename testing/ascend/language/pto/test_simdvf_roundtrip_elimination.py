"""Structural contract tests for ``EliminateSimdVFRoundTrip``.

Hardware independent: each case lowers TileLang IR for the PTO target with
:func:`tilelang.engine.lower.lower` and inspects the module the new pass
produces (and the module ``VectorizeParallelToPTO`` produces from it). No
PTOAS/Bisheng device compile and no device execution happen here; the numeric
counterpart lives in ``test_simdvf_roundtrip_elimination_run.py``.

The pass removes the UB round-trip buffers that a SIMD_VF region uses only to
shuttle a value between units (``a[i] = a_ub[i]`` ... ``y_ub[i] = y[i]``).
After vectorization such a buffer becomes a vstore followed by a vload in the
same vecscope, which the machine-level schedule does not order as a
dependency. The contract asserted here is the *absence* of those buffers and
accesses, plus the untouched shape of every case the rules must not rewrite:

* a loop-carried copy under ``T.serial`` (cf2v's ``scores``) is not a
  top-level unit pair and must stay;
* a buffer read outside the region is not region-local and must stay;
* a buffer with two writers, mismatched unit extents, or a tail extent is
  outside the whitelist and must stay.

The theoretical "region becomes empty" case is unreachable with these rules
(each rewrite removes one unit and always keeps a writing unit), so the pass
only carries the no-op body rebuild defensively; ``test_region_without_units``
covers the neighbouring path (a SIMD_VF region with no ``T.Parallel`` at all,
which both this pass and Vectorize skip).
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

import tilelang  # noqa: F401
import tilelang.ascend.language as T
from tilelang.ascend import transform as ascend_transform
from tilelang.ascend.target import normalize_pto_target
from tilelang.engine.lower import lower as _lower
from tvm import tirx
from tvm.tirx.stmt_functor import post_order_visit

E, K, LANES = 256, 8, 64
EPS = 1e-4
NEG = float(-3.402823e38)


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


def _cf5v():
    """cf5v: copy in, compare/select, copy out - two round-trip buffers."""

    @T.prim_func
    def main(A: T.Tensor((E,), "float32"), B: T.Tensor((E,), "float32"),
             Y: T.Tensor((E,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((E,), "float32")
            b_ub = T.alloc_shared((E,), "float32")
            y_ub = T.alloc_shared((E,), "float32")
            quot = T.alloc_shared((E,), "float32")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            for i in T.serial(E):
                quot[i] = a_ub[i] / b_ub[i]
            with T.SimdVF(lanes=LANES):
                a = T.alloc_shared((E,), "float32")
                y = T.alloc_shared((E,), "float32")
                for i in T.Parallel(E):
                    a[i] = a_ub[i]
                for i in T.Parallel(E):
                    ax = T.Select(a[i] >= T.float32(0.0), a[i],
                                  a[i] * T.float32(-1.0))
                    y[i] = T.Select(ax < EPS, T.float32(0.0), quot[i])
                for i in T.Parallel(E):
                    y_ub[i] = y[i]
            T.copy(y_ub, Y)

    return main


def _cf2v():
    """cf2v: the copy/consume pair sits under ``for k in T.serial(K)``."""

    @T.prim_func
    def main(A: T.Tensor((E,), "float32"), Thresh: T.Tensor((K,), "float32"),
             Out: T.Tensor((E,), "float32")):
        with T.Kernel(1):
            s = T.alloc_shared((E,), "float32")
            thr_ub = T.alloc_shared((K,), "float32")
            out_ub = T.alloc_shared((E,), "float32")
            T.copy(A, s)
            T.copy(Thresh, thr_ub)
            with T.SimdVF(lanes=LANES):
                scores = T.alloc_shared((E,), "float32")
                for k in T.serial(K):
                    thr = thr_ub[k]
                    for i in T.Parallel(E):
                        scores[i] = s[i]
                    for i in T.Parallel(E):
                        s[i] = T.Select(scores[i] > thr, T.float32(NEG), s[i])
                for i in T.Parallel(E):
                    out_ub[i] = s[i]
            T.copy(out_ub, Out)

    return main


def _roundtrip_with_outside_reader():
    """The round-trip buffer is allocated outside the region and read after it."""

    @T.prim_func
    def main(A: T.Tensor((E,), "float32"), Y: T.Tensor((E,), "float32"),
             Z: T.Tensor((E,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((E,), "float32")
            y_ub = T.alloc_shared((E,), "float32")
            tmp = T.alloc_shared((E,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=LANES):
                for i in T.Parallel(E):
                    tmp[i] = a_ub[i]
                for i in T.Parallel(E):
                    y_ub[i] = tmp[i]
            T.copy(y_ub, Y)
            T.copy(tmp, Z)

    return main


def _roundtrip_multi_writer():
    """Two units write the round-trip buffer."""

    @T.prim_func
    def main(A: T.Tensor((E,), "float32"), B: T.Tensor((E,), "float32"),
             Y: T.Tensor((E,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((E,), "float32")
            b_ub = T.alloc_shared((E,), "float32")
            y_ub = T.alloc_shared((E,), "float32")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF(lanes=LANES):
                tmp = T.alloc_shared((E,), "float32")
                for i in T.Parallel(E):
                    tmp[i] = a_ub[i]
                for i in T.Parallel(E):
                    tmp[i] = b_ub[i]
                for i in T.Parallel(E):
                    y_ub[i] = tmp[i]
            T.copy(y_ub, Y)

    return main


def _roundtrip_extent_mismatch():
    """Producer and consumer units have different extents."""

    @T.prim_func
    def main(A: T.Tensor((E,), "float32"), Y: T.Tensor((E,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((E,), "float32")
            y_ub = T.alloc_shared((E,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=LANES):
                tmp = T.alloc_shared((E,), "float32")
                for i in T.Parallel(E // 2):
                    tmp[i] = a_ub[i]
                for i in T.Parallel(E):
                    y_ub[i] = tmp[i]
            T.copy(y_ub, Y)

    return main


def _roundtrip_tail():
    """A non-divisible extent needs a tail mask and stays untouched."""

    @T.prim_func
    def main(A: T.Tensor((256,), "float32"), Y: T.Tensor((256,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((256,), "float32")
            y_ub = T.alloc_shared((256,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=LANES):
                tmp = T.alloc_shared((256,), "float32")
                for i in T.Parallel(200):
                    tmp[i] = a_ub[i]
                for i in T.Parallel(200):
                    y_ub[i] = tmp[i]
            T.copy(y_ub, Y)

    return main


def _region_without_units():
    """A hand-written region with no T.Parallel is skipped by both passes."""

    @T.prim_func
    def main(A: T.Tensor((E,), "float32"), Y: T.Tensor((E,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((E,), "float32")
            y_ub = T.alloc_shared((E,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=LANES):
                for i in T.serial(E):
                    y_ub[i] = a_ub[i]
            T.copy(y_ub, Y)

    return main


# ---------------------------------------------------------------------------
# lowering / inspection helpers
# ---------------------------------------------------------------------------


@contextmanager
def _stage_capture():
    """Capture the module the new pass receives and the one it returns."""
    captured = {}
    original = ascend_transform.EliminateSimdVFRoundTrip

    def capturing_factory():
        inner = original()

        def run(mod):
            captured["input"] = mod
            result = inner(mod)
            captured["output"] = result
            return result

        return run

    ascend_transform.EliminateSimdVFRoundTrip = capturing_factory
    try:
        yield captured
    finally:
        ascend_transform.EliminateSimdVFRoundTrip = original


def _run_pass(prim):
    """Lower ``prim`` for PTO and return (pass input, pass output)."""
    with _stage_capture() as captured:
        target = normalize_pto_target("pto")
        with target:
            _lower(prim, target=target)
    return captured["input"], captured["output"]


def _simd_block(mod):
    blocks = []

    def visit(node):
        if isinstance(node, tirx.SBlock) and node.name_hint == "SIMD_VF":
            blocks.append(node)

    post_order_visit(mod["main"].body, visit)
    assert len(blocks) == 1, f"expected one SIMD_VF block, got {len(blocks)}"
    return blocks[0]


def _top_level_stmts(simd):
    body = simd.body
    while isinstance(body, tirx.AttrStmt):
        body = body.body
    if isinstance(body, tirx.SeqStmt):
        return list(body.seq)
    return [body]


def _parallel_units(simd):
    return [
        stmt for stmt in _top_level_stmts(simd)
        if isinstance(stmt, tirx.For) and stmt.kind == tirx.ForKind.PARALLEL
    ]


def _accesses(stmt):
    found = []

    def visit(node):
        if isinstance(node, tirx.BufferLoad):
            found.append(("load", node.buffer.name))
        elif isinstance(node, tirx.BufferStore):
            found.append(("store", node.buffer.name))

    post_order_visit(stmt, visit)
    return found


def _access_ptr_roots(stmt):
    """Buffer names that appear as the base of a tl.access_ptr call."""
    roots = []

    def visit(node):
        if isinstance(node, tirx.Call) and node.op.name == "tl.access_ptr":
            buf = node.args[0]
            if isinstance(buf, tirx.BufferLoad):
                roots.append(buf.buffer.name)

    post_order_visit(stmt, visit)
    return roots


# ---------------------------------------------------------------------------
# tests: the rewrite
# ---------------------------------------------------------------------------


def test_cf5v_roundtrip_fully_eliminated():
    _, output = _run_pass(_cf5v())
    simd = _simd_block(output)

    assert [b.name for b in simd.alloc_buffers] == []
    units = _parallel_units(simd)
    assert len(units) == 1, "both round-trip units must be gone"

    accesses = _accesses(units[0])
    assert ("store", "y_ub") in accesses
    assert ("load", "a_ub") in accesses
    assert ("load", "quot") in accesses
    assert not [name for _, name in accesses if name in ("a", "y")]


def test_cf5v_vectorized_single_chunk_loop():
    captured = {}
    target = normalize_pto_target("pto")

    original_eliminate = ascend_transform.EliminateSimdVFRoundTrip
    original_vectorize = ascend_transform.VectorizeParallelToPTO

    def eliminate_factory():
        inner = original_eliminate()

        def run(mod):
            result = inner(mod)
            captured["eliminate"] = result
            return result

        return run

    def vectorize_factory():
        inner = original_vectorize()

        def run(mod):
            result = inner(mod)
            captured["vectorize"] = result
            return result

        return run

    ascend_transform.EliminateSimdVFRoundTrip = eliminate_factory
    ascend_transform.VectorizeParallelToPTO = vectorize_factory
    try:
        with target:
            _lower(_cf5v(), target=target)
    finally:
        ascend_transform.EliminateSimdVFRoundTrip = original_eliminate
        ascend_transform.VectorizeParallelToPTO = original_vectorize

    text = str(captured["vectorize"])
    # One chunk loop for the only remaining unit, one vstore writing y_ub.
    assert text.count("for q in range(4)") == 1
    assert text.count("T.tl.vmi.vstore") == 1
    simd = _simd_block(captured["vectorize"])
    assert simd is not None
    roots = _access_ptr_roots(simd.body)
    assert "y_ub" in roots
    assert "a" not in roots and "y" not in roots
    # The S -> V edge that publishes `quot` must survive.
    assert 'T.ascend_wait_flag("S_V", 0)' in text


def test_cf2v_loop_carried_untouched():
    captured = {}
    target = normalize_pto_target("pto")
    original_eliminate = ascend_transform.EliminateSimdVFRoundTrip

    def eliminate_factory():
        inner = original_eliminate()

        def run(mod):
            captured["input"] = mod
            result = inner(mod)
            captured["output"] = result
            return result

        return run

    ascend_transform.EliminateSimdVFRoundTrip = eliminate_factory
    try:
        with target:
            _lower(_cf2v(), target=target)
    finally:
        ascend_transform.EliminateSimdVFRoundTrip = original_eliminate

    simd = _simd_block(captured["output"])
    assert [b.name for b in simd.alloc_buffers] == ["scores"]

    top = _top_level_stmts(simd)
    assert len(top) == 2
    assert isinstance(top[0], tirx.For) and top[0].kind == tirx.ForKind.SERIAL

    accesses = _accesses(simd.body)
    assert ("store", "scores") in accesses
    assert ("load", "scores") in accesses
    # The serial-carried value keeps its UB form as well.
    assert any(name == "s" and kind == "store" for kind, name in accesses)


def test_idempotent():
    _, output = _run_pass(_cf5v())
    second = ascend_transform.EliminateSimdVFRoundTrip()(output)
    assert str(second) == str(output)


# ---------------------------------------------------------------------------
# tests: conservative skips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("builder", [_cf5v, _cf2v])
def test_result_still_passes_verify(builder):
    _, output = _run_pass(builder())
    # Must not throw: the rewritten module stays inside the Verify contract.
    ascend_transform.VerifyParallelToPTO()(output)


def test_negative_outside_reader():
    # The buffer is allocated outside the region (so it is not a region-local
    # allocation) and is also read after the region: both facts must keep it.
    _, output = _run_pass(_roundtrip_with_outside_reader())
    simd = _simd_block(output)

    accesses = _accesses(simd.body)
    assert ("store", "tmp") in accesses
    assert ("load", "tmp") in accesses
    assert len(_parallel_units(simd)) == 2


def test_negative_multi_writer():
    _, output = _run_pass(_roundtrip_multi_writer())
    simd = _simd_block(output)
    assert "tmp" in [b.name for b in simd.alloc_buffers]
    assert len(_parallel_units(simd)) == 3


def test_negative_extent_mismatch():
    _, output = _run_pass(_roundtrip_extent_mismatch())
    simd = _simd_block(output)
    assert "tmp" in [b.name for b in simd.alloc_buffers]
    assert len(_parallel_units(simd)) == 2


def test_negative_tail_unit():
    _, output = _run_pass(_roundtrip_tail())
    simd = _simd_block(output)
    assert "tmp" in [b.name for b in simd.alloc_buffers]
    assert len(_parallel_units(simd)) == 2


def test_region_without_units():
    # No T.Parallel: the pass must leave the region alone and lowering must
    # still succeed (Vectorize skips it).
    captured = {}
    target = normalize_pto_target("pto")
    original = ascend_transform.EliminateSimdVFRoundTrip

    def factory():
        inner = original()

        def run(mod):
            captured["output"] = inner(mod)
            return captured["output"]

        return run

    ascend_transform.EliminateSimdVFRoundTrip = factory
    try:
        with target:
            _lower(_region_without_units(), target=target)
    finally:
        ascend_transform.EliminateSimdVFRoundTrip = original

    simd = _simd_block(captured["output"])
    assert _parallel_units(simd) == []
    assert [b.name for b in simd.alloc_buffers] == []
