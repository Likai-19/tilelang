"""PTO Parallel control-flow tests.

Hardware independent: each case lowers TileLang IR for the PTO target with
:func:`tilelang.engine.lower.lower` and inspects the module produced by
``LegalizeParallelToPTO`` / the final lowering (no PTOAS/Bisheng device
compile, no device execution unless the case is explicitly a numeric one).

Expected contract:

* a lane-varying ``if`` / ``if``-``else`` under ``T.Parallel`` is rewritten by
  Legalize into the safe scalar ``Select`` form: stores are paired by written
  address, and a store without a counterpart in the other arm keeps the old
  value (``Select(cond, new, old_value)`` — the same form a single-sided
  ``if`` produces);
* comparisons, ``T.min``/``T.max`` and boolean operators follow the same
  admission rules and end as ``vcmp``/``vsel`` (or are rejected with a
  diagnostic when the source semantics cannot be preserved);
* the frontend ``-T.infinity()`` sentinel is rewritten to a ``FloatImm``
  constant, so no frontend op name survives Legalize;
* explicit padding (``T.Parallel(P)`` with ``if i < E ... else ...``) keeps its
  ``else`` branch and legalizes to the Select form;
* Legalize itself never emits VMI and never widens a conditional load: the
  runtime address safety of full-width/masked loads stays the caller's
  contract, and a lowering that cannot preserve the source condition is
  rejected with a diagnostic.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

import tilelang.ascend.language as T
from tilelang.ascend import transform as ascend_transform
from tilelang.ascend.target import normalize_pto_target
from tilelang.engine.lower import lower as _lower
import tilelang
import tvm


@contextmanager
def _stage_capture():
    """Capture the modules around Legalize and after Vectorize.

    ``captured["legalize"]`` is the module LegalizeParallelToPTO returned
    (still scalar parallel TIR: Select form, no VMI yet);
    ``captured["vectorize"]`` is the module VectorizeParallelToPTO returned
    (the VMI form).
    """
    captured = {}
    original_legalize = ascend_transform.LegalizeParallelToPTO
    original_vectorize = ascend_transform.VectorizeParallelToPTO

    def capturing_legalize_factory():
        legalize_pass = original_legalize()

        def run(mod):
            result = legalize_pass(mod)
            captured["legalize"] = result
            return result

        return run

    def capturing_vectorize_factory():
        vectorize_pass = original_vectorize()

        def run(mod):
            result = vectorize_pass(mod)
            captured["vectorize"] = result
            return result

        return run

    ascend_transform.LegalizeParallelToPTO = capturing_legalize_factory
    ascend_transform.VectorizeParallelToPTO = capturing_vectorize_factory
    try:
        yield captured
    finally:
        ascend_transform.LegalizeParallelToPTO = original_legalize
        ascend_transform.VectorizeParallelToPTO = original_vectorize


def _lower_capture(func, *, expect_error=False):
    """Lower *func* for PTO and return (captured_stages, error)."""
    target = normalize_pto_target("pto")
    error = None
    with _stage_capture() as captured:
        try:
            with target:
                _lower(func, target=target)
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            error = exc
    if error is not None and not expect_error:
        raise error
    return captured, error


def _legalize_direct(func):
    """Run Legalize directly so front-end Simplify cannot erase scalar Min/Max."""
    target = normalize_pto_target("pto")
    mod = tvm.IRModule.from_expr(func.with_attr("target", target))
    with target:
        return ascend_transform.LegalizeParallelToPTO()(mod)


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


def _expr_loads(expr):
    """Buffer names read by an expression (no module wrapper)."""
    found = []

    def visit(node):
        if isinstance(node, tvm.tirx.BufferLoad):
            found.append(node.buffer.name)

    tvm.tirx.stmt_functor.post_order_visit(expr, visit)
    return set(found)


def _vmi_call_names(mod):
    names = {}

    def visit(node):
        if isinstance(node, tvm.tirx.Call):
            op = getattr(node, "op", None)
            name = getattr(op, "name", None)
            if isinstance(name, str) and name.startswith("tl.vmi."):
                names[name] = names.get(name, 0) + 1

    tvm.tirx.stmt_functor.post_order_visit(_body_of(mod), visit)
    return names


def _selects(mod):
    return _collect(mod, lambda n: isinstance(n, tvm.tirx.Select))


def _if_then_elses(mod):
    return _collect(mod, lambda n: isinstance(n, tvm.tirx.IfThenElse))


def _max_min_nodes(mod):
    return _collect(
        mod,
        lambda n: isinstance(n, (tvm.tirx.Max, tvm.tirx.Min)),
    )


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def _k_if_else(E, lanes, dtype="float32"):
    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(
        A: T.Tensor((P,), dtype),
        B_init: T.Tensor((P,), dtype),
        B: T.Tensor((P,), dtype),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((P,), dtype)
            b = T.alloc_shared((P,), dtype)
            T.copy(A, a)
            T.copy(B_init, b)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    if a[i] > 0:
                        b[i] = a[i] * 2
                    else:
                        b[i] = a[i] + 1
            T.copy(b, B)

    return main


def _k_single_sided(E, lanes, dtype="float32"):
    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(
        A: T.Tensor((P,), dtype),
        B_init: T.Tensor((P,), dtype),
        B: T.Tensor((P,), dtype),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((P,), dtype)
            b = T.alloc_shared((P,), dtype)
            T.copy(A, a)
            T.copy(B_init, b)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    if a[i] > 0:
                        b[i] = a[i] * 2
            T.copy(b, B)

    return main


def _k_minmax(E, lanes, dtype="float32"):
    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(
        A: T.Tensor((P,), dtype),
        B: T.Tensor((P,), dtype),
        C: T.Tensor((P,), dtype),
        D: T.Tensor((P,), dtype),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((P,), dtype)
            b = T.alloc_shared((P,), dtype)
            c = T.alloc_shared((P,), dtype)
            d = T.alloc_shared((P,), dtype)
            T.copy(A, a)
            T.copy(B, b)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    c[i] = T.max(a[i], b[i])
                    d[i] = T.min(a[i], b[i])
            T.copy(c, C)
            T.copy(d, D)

    return main


def _k_explicit_padding(E, lanes, dtype="float32"):
    P = (E + lanes - 1) // lanes * lanes

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
def test_if_else_legalizes_to_select_without_residual_branch():
    E, lanes = 150, 128
    stages, _ = _lower_capture(_k_if_else(E, lanes))

    legalized = stages["legalize"]
    assert not _if_then_elses(legalized), "Legalize must remove the lane-varying IfThenElse"
    assert _selects(legalized), "the both-arms store must become a Select"

    vectorized = stages["vectorize"]
    names = _vmi_call_names(vectorized)
    assert "tl.vmi.vcmp" in names or "tl.vmi.vcmps" in names, "the comparison must lower to vcmp/vcmps"
    assert "tl.vmi.vsel" in names, "the Select must lower to vsel"
    assert "tl.vmi.vand" not in names, "this case must not need the boolean combine"

    # A conditional nested inside a branch recurses through the same rewrite
    # and ends as nested Selects (no residual IfThenElse anywhere).
    nested, _ = _lower_capture(_k_nested_three_way())
    nested_legalized = nested["legalize"]
    assert not _if_then_elses(nested_legalized), "nested conditionals must be legalized away"
    assert len(_selects(nested_legalized)) >= 2, "the nested predicates must become nested Selects"
    nested_names = _vmi_call_names(nested["vectorize"])
    assert nested_names.get("tl.vmi.vcmp", 0) >= 2, f"both comparisons must lower to vcmp, got {nested_names}"
    assert "tl.vmi.vsel" in nested_names, "the Selects must lower to vsel"


def _predicated_stores(stmt):
    return [
        n
        for n in _collect(stmt, lambda n: isinstance(n, tvm.tirx.BufferStore))
        if n.predicate is not None
    ]


def _stores(stmt):
    return _collect(stmt, lambda n: isinstance(n, tvm.tirx.BufferStore))


@pytest.mark.pto
def test_single_sided_if_reads_back_the_destination():
    """A one-sided branch keeps the destination's old value on the lanes the
    condition does not select, so Legalize folds it into a read-back Select:
    the value choice loads the destination itself and the store stays
    unpredicated. Vectorize lowers it to vsel plus a full-mask vstore."""
    E, lanes = 150, 128
    stages, _ = _lower_capture(_k_single_sided(E, lanes))

    legalized = stages["legalize"]
    assert not _if_then_elses(legalized), "Legalize must remove the single-sided IfThenElse"

    stores = _stores(legalized)
    assert len(stores) == 1, f"the one-sided branch must be one store, got {len(stores)}"
    store = stores[0]
    assert store.buffer.name == "b"
    assert store.predicate is None, "a read-back store must not carry a predicate"
    assert _selects(legalized), "the arm must become a value choice"
    assert "b" in _expr_loads(store.value), (
        "the value choice must read the destination's old value back, got "
        f"{sorted(_expr_loads(store.value))}"
    )

    names = _vmi_call_names(stages["vectorize"])
    assert "tl.vmi.vsel" in names, "the value choice must lower to vsel"
    assert "tl.vmi.vstore" in names, "the store must survive"
    assert "tl.vmi.vand" not in names, "no predicate is conjoined into the store mask"


@pytest.mark.pto
def test_conditional_arm_that_reads_its_own_write_keeps_arm_order():
    """A store may read what an earlier store of the same arm wrote. The two
    arms are then emitted in their source order, each gated by the branch
    condition, so the read-after-write keeps its meaning; the backend's
    vecscope memory barriers order the resulting UB accesses."""

    E, lanes = 150, 128
    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(A: T.Tensor((P,), "float32"), Y: T.Tensor((2, P), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((P,), "float32")
            y = T.alloc_shared((2, P), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    if a[i] > 0:
                        y[0, i] = a[i]
                        y[1, i] = y[0, i] * 2
                    else:
                        y[1, i] = a[i]
                        y[0, i] = y[1, i] * 3
            T.copy(y, Y)

    mod, error = _lower_capture(main, expect_error=True)
    assert error is None, f"the ordered arm read-after-write must lower, got: {error}"

    legalized = mod["legalize"]
    assert not _if_then_elses(legalized), "the branch must be legalized away"
    stores = _stores(legalized)
    # Two stores per arm, emitted in source order; each keeps its own
    # destination's old value through a read-back value choice.
    assert len(stores) == 4, f"expected four stores, got {len(stores)}"
    assert all(st.predicate is None for st in stores), "no store may stay predicated"
    order = [st.indices[0] for st in stores]
    assert [int(str(idx)) for idx in order] == [0, 1, 1, 0], (
        f"the arm order must be kept, got row indices {order}"
    )
    assert len(_selects(legalized)) >= 4, "every store needs its own value choice"


def test_minmax_legalizes_without_residual_max_min_nodes():
    E, lanes = 150, 128
    stages, _ = _lower_capture(_k_minmax(E, lanes))

    legalized = stages["legalize"]
    assert not _max_min_nodes(legalized), "Legalize must rewrite Min/Max into the Select form"
    assert _selects(legalized), "the rewritten Min/Max must be a Select before vectorization"

    vectorized = stages["vectorize"]
    names = _vmi_call_names(vectorized)
    assert "tl.vmi.vsel" in names, "the rewritten Min/Max must lower to vsel"


@pytest.mark.pto
def test_explicit_padding_keeps_its_else_branch():
    E, lanes = 150, 128
    stages, _ = _lower_capture(_k_explicit_padding(E, lanes))

    vectorized = stages["vectorize"]
    names = _vmi_call_names(vectorized)
    assert "tl.vmi.vcmp" in names or "tl.vmi.vcmps" in names, "the guard must lower to a comparison"
    assert "tl.vmi.vsel" in names, "the preserved else must lower to vsel"
    assert "tl.vmi.vstore" in names, "the guarded store must survive as a masked vstore"
    # The guard covers [0, E) of the user's P domain: the vstore mask is the
    # full mask, never a remaining-based prefix mask.
    alloc_names = {node.buffer.name for node in _collect(vectorized, lambda n: isinstance(n, tvm.tirx.AllocBuffer))}
    assert "remaining" not in alloc_names, f"explicit padding must not need a tail mask: {alloc_names}"


@pytest.mark.pto
def test_conditional_load_pre_evaluates_under_the_caller_contract():
    """A lane-varying guard around a load is admitted: Legalize keeps the load
    inside a Select arm (both arms pre-evaluated); the runtime readability of
    that full-width access is the input program's memory contract, not
    something the compiler proves."""

    E, lanes = 150, 128
    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(
        A: T.Tensor((P,), "float32"),
        B: T.Tensor((P,), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((P,), "float32")
            b = T.alloc_shared((P,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    if i < E - 1:
                        b[i] = a[i + 1]
                    else:
                        b[i] = a[i]
            T.copy(b, B)

    stages, _ = _lower_capture(main)
    legalized = stages["legalize"]
    assert not _if_then_elses(legalized), "the guard must be legalized away"
    selects = _selects(legalized)
    assert selects, "the guarded load must be preserved inside a Select arm"

    loads = _collect(legalized, lambda n: isinstance(n, tvm.tirx.BufferLoad))
    assert loads, "the guarded load must survive (pre-evaluated arm), not be dropped"


@pytest.mark.pto
def test_divergent_branch_destinations_read_back_their_destination():
    """Arms that write different destinations each keep their own buffer's
    old value on the lanes the condition blocks: each destination gets its own
    read-back value choice (unpredicated store)."""

    E, lanes = 150, 128
    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(
        A: T.Tensor((P,), "float32"),
        B_init: T.Tensor((P,), "float32"),
        C_init: T.Tensor((P,), "float32"),
        B: T.Tensor((P,), "float32"),
        C: T.Tensor((P,), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((P,), "float32")
            b = T.alloc_shared((P,), "float32")
            c = T.alloc_shared((P,), "float32")
            T.copy(A, a)
            T.copy(B_init, b)
            T.copy(C_init, c)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    if a[i] > 0:
                        b[i] = a[i] * 2
                    else:
                        c[i] = a[i] + 1
            T.copy(b, B)
            T.copy(c, C)

    stages, error = _lower_capture(main)
    assert error is None, f"divergent destinations must lower: {error}"
    legalized = stages["legalize"]
    assert not _if_then_elses(legalized), "the branch must be legalized away"

    stores = _stores(legalized)
    assert len(stores) == 2, f"each destination gets its own store, got {len(stores)}"
    assert {s.buffer.name for s in stores} == {"b", "c"}
    for store in stores:
        # Each store keeps its own destination's old value: the value choice
        # reads that destination back, and the store stays unpredicated.
        assert store.predicate is None, f"store into {store.buffer.name} must not be predicated"
        assert store.buffer.name in _expr_loads(store.value), (
            f"the value choice for {store.buffer.name} must read it back, got "
            f"{sorted(_expr_loads(store.value))}"
        )
    assert len(_selects(legalized)) >= 2, "each destination needs its own value choice"


@pytest.mark.pto
def test_infinity_constant_legalizes_to_floatimm():
    """The frontend ``-T.infinity()`` sentinel is rewritten to a FloatImm
    constant by Legalize; Vectorize's constant handling covers it and no
    frontend op name survives into the vectorized IR."""

    E, lanes = 150, 128
    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(
        A: T.Tensor((P,), "float32"),
        B: T.Tensor((P,), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((P,), "float32")
            b = T.alloc_shared((P,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    if a[i] > 0:
                        b[i] = a[i]
                    else:
                        b[i] = -T.infinity("float32")
            T.copy(b, B)

    stages, error = _lower_capture(main)
    assert error is None, f"infinity must lower as a constant: {error}"
    for stage in ("legalize", "vectorize"):
        text = _body_of(stages[stage]).__str__()
        assert "infinity" not in text, (
            f"the frontend infinity op must not survive {stage}"
        )
    names = _vmi_call_names(stages["vectorize"])
    assert names.get("tl.vmi.vsel") == 1, f"the Select must lower to vsel, got {names}"
    assert names.get("tl.vmi.vbrc", 0) >= 1, "the constant must reach a vbrc"


def _k_nested_three_way(E=150, lanes=128, dtype="float32"):
    """Three-way classify: nested if inside the then-arm, plain else."""

    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(
        A: T.Tensor((P,), dtype),
        B: T.Tensor((P,), dtype),
        C: T.Tensor((P,), dtype),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((P,), dtype)
            b = T.alloc_shared((P,), dtype)
            c = T.alloc_shared((P,), dtype)
            T.copy(A, a)
            T.copy(B, b)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    if a[i] > 0:
                        if b[i] > 0:
                            c[i] = a[i] * 2
                        else:
                            c[i] = b[i] * 2
                    else:
                        c[i] = a[i] + b[i]
            T.copy(c, C)

    return main


def _k_arm_read_then_write(E=150, lanes=128, dtype="float32"):
    """One arm reads a buffer and then overwrites it (WAR); the other writes
    and then reads it back (RAW). Both need the arm's own order preserved."""

    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(A: T.Tensor((P,), dtype), Y: T.Tensor((2, P), dtype)):
        with T.Kernel(1):
            a = T.alloc_shared((P,), dtype)
            y = T.alloc_shared((2, P), dtype)
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    y[0, i] = T.float32(7.0)
                for i in T.Parallel(E):
                    if a[i] > T.float32(0.0):
                        y[1, i] = y[0, i]
                        y[0, i] = T.float32(1.0)
                    else:
                        y[0, i] = T.float32(2.0)
                        y[1, i] = y[0, i]
            T.copy(y, Y)

    return main


def _k_scalar_bool_select(E=150, lanes=128, dtype="float32"):
    """Both arms write one address and the condition is a bare scalar bool."""

    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(
        A: T.Tensor((P,), dtype),
        B: T.Tensor((P,), dtype),
        flag: T.bool,
        Y: T.Tensor((P,), dtype),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((P,), dtype)
            b = T.alloc_shared((P,), dtype)
            y = T.alloc_shared((P,), dtype)
            T.copy(A, a)
            T.copy(B, b)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    if flag:
                        y[i] = a[i]
                    else:
                        y[i] = b[i]
            T.copy(y, Y)

    return main


def _k_scalar_bool_predicate(E=150, lanes=128, dtype="float32"):
    """One-sided branch under a bare scalar bool."""

    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(
        A: T.Tensor((P,), dtype),
        flag: T.bool,
        Y: T.Tensor((P,), dtype),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((P,), dtype)
            y = T.alloc_shared((P,), dtype)
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    if flag:
                        y[i] = a[i]
            T.copy(y, Y)

    return main


def _k_scalar_bool_and_lane_cond(E=150, lanes=128, dtype="float32"):
    """A scalar bool combined with a lane-varying condition."""

    P = (E + lanes - 1) // lanes * lanes

    @T.prim_func
    def main(
        A: T.Tensor((P,), dtype),
        flag: T.bool,
        Y: T.Tensor((P,), dtype),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((P,), dtype)
            y = T.alloc_shared((P,), dtype)
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    if flag and a[i] > T.float32(0.0):
                        y[i] = a[i]
            T.copy(y, Y)

    return main


def _k_boolean_mask(E, lanes, dtype="float32"):
    """A boolean expression over two comparisons used as the guard; the mask
    is bound first (a value position), so the condition keeps the mask-logic
    form through Legalize and Vectorize converts it to vand."""

    @T.prim_func
    def main(
        A: T.Tensor((E,), dtype),
        B: T.Tensor((E,), dtype),
        C: T.Tensor((E,), dtype),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((E,), dtype)
            b = T.alloc_shared((E,), dtype)
            c = T.alloc_shared((E,), dtype)
            T.copy(A, a)
            T.copy(B, b)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    m = (a[i] > 0) & (b[i] > 0)
                    if m:
                        c[i] = a[i]
                    else:
                        c[i] = b[i]
            T.copy(c, C)

    return main


@pytest.mark.pto
def test_conditional_arm_with_read_then_write_keeps_arm_order():
    """A WAR inside one arm (read a buffer, then overwrite it) is accepted: both
    arms are emitted in source order with their branch predicate, and the
    backend orders the resulting accesses (mem_bar)."""

    stages, error = _lower_capture(_k_arm_read_then_write())
    assert error is None, f"WAR/RAW inside an arm must lower, got: {error}"

    legalized = stages["legalize"]
    assert not _if_then_elses(legalized), "the branch must be legalized away"
    stores = _stores(legalized)
    # One unconditional init store plus the four branch stores, each of which
    # keeps its destination's old value through a read-back value choice.
    assert len(stores) == 5, f"expected init + four read-back stores, got {len(stores)}"
    readback = [st for st in stores if isinstance(st.value, tvm.tirx.Select)]
    assert len(readback) == 4, f"expected four read-back stores, got {len(readback)}"
    assert all(st.predicate is None for st in stores), "no store may stay predicated"


@pytest.mark.pto
def test_scalar_bool_select_condition_becomes_a_full_mask():
    """A bare scalar bool has no operand to compare, so it becomes the active
    lane count of a second `create_mask`: all lanes for true, none for false.
    The result is an all-true / all-false lane mask that `vsel` can consume."""

    stages, error = _lower_capture(_k_scalar_bool_select())
    assert error is None, f"a scalar bool condition must lower, got: {error}"

    legalized = stages["legalize"]
    assert not _if_then_elses(legalized), "the branch must be legalized away"
    assert _selects(legalized), "both arms write one address: a Select"

    names = _vmi_call_names(stages["vectorize"])
    assert names.get("tl.vmi.create_mask", 0) >= 2, (
        f"the scalar bool must produce its own create_mask, got {names}"
    )
    assert "tl.vmi.vcmps" not in names, (
        f"a scalar bool needs no vector compare carrier, got {names}"
    )
    assert "tl.vmi.vsel" in names, f"the Select must lower to vsel, got {names}"


@pytest.mark.pto
def test_scalar_bool_condition_of_a_one_sided_branch_is_widened():
    """A bare scalar bool condition is widened to a lane mask before it feeds
    the read-back value choice: the flag becomes the active lane count of a
    `create_mask`, giving an all-true / all-false mask consumed by vsel."""

    stages, error = _lower_capture(_k_scalar_bool_predicate())
    assert error is None, f"a scalar bool condition must lower, got: {error}"

    legalized = stages["legalize"]
    stores = _stores(legalized)
    assert len(stores) == 1, f"expected one store, got {len(stores)}"
    assert stores[0].predicate is None, "the read-back store must not be predicated"
    assert _selects(legalized), "the arm must become a value choice"

    names = _vmi_call_names(stages["vectorize"])
    assert names.get("tl.vmi.create_mask", 0) >= 2, (
        f"the scalar bool must become its own mask, got {names}"
    )
    assert "tl.vmi.vcmps" not in names, (
        f"a scalar bool needs no vector compare carrier, got {names}"
    )
    assert "tl.vmi.vsel" in names, "the value choice must lower to vsel"
    assert "tl.vmi.vstore" in names, "the store must survive"


@pytest.mark.pto
def test_scalar_bool_in_a_conjunction_is_widened_before_the_combine():
    """`flag and (a[i] > 0)` is distributed into nested value choices: the
    scalar side becomes its own `create_mask` first, so the outer choice
    consumes a real mask. Only the lane comparison needs a vector compare."""

    stages, error = _lower_capture(_k_scalar_bool_and_lane_cond())
    assert error is None, f"a scalar bool in a conjunction must lower, got: {error}"

    names = _vmi_call_names(stages["vectorize"])
    assert names.get("tl.vmi.create_mask", 0) >= 2, (
        f"the scalar bool must be widened first, got {names}"
    )
    assert "tl.vmi.vcmp" in names or "tl.vmi.vcmps" in names, (
        f"the lane comparison still needs a vector compare, got {names}"
    )
    assert names.get("tl.vmi.vsel", 0) >= 2, (
        f"the conjunction must become nested value choices, got {names}"
    )


@pytest.mark.pto
def test_boolean_mask_value_lowers_to_vand():
    E, lanes = 256, 128
    stages, _ = _lower_capture(_k_boolean_mask(E, lanes))

    names = _vmi_call_names(stages["vectorize"])
    assert names.get("tl.vmi.vand") == 1, f"vand must be emitted once, got {names}"
    assert "tl.vmi.vsel" in names, "the guarded Select must lower to vsel"
    assert names.get("tl.vmi.vcmp", 0) >= 2, "both comparisons must lower to vcmp"

    # The boolean combine is the existing PTODSL bitwise vector op on mask
    # registers, so the emitted PTODSL name equals the TileLang op name.
    kernel = tilelang.compile(_k_boolean_mask(E, lanes), target="pto", out_idx=-1)
    source = kernel.get_kernel_source()
    assert "pto.vmi.vand(" in source, "the boolean combine must emit pto.vmi.vand"


@pytest.mark.pto
def test_lane_uniform_store_is_still_rejected():
    """A store whose address does not vary with the lane stays out of scope:
    Verify rejects it and Legalize never materialises a reducer for it."""

    E, lanes = 128, 128

    @T.prim_func
    def main(
        A: T.Tensor((E,), "float32"),
        B: T.Tensor((E,), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((E,), "float32")
            b = T.alloc_shared((E,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    if i < E // 2:
                        b[0] = a[i]
            T.copy(b, B)

    _, error = _lower_capture(main, expect_error=True)
    assert error is not None, "a lane-uniform store must stay rejected at this stage"
    message = str(error)
    assert "lane-uniform" in message, f"expected the lane-uniform rejection, got: {message}"
