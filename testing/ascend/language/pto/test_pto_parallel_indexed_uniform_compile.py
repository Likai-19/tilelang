"""Compile duplicate uniform scatter targets without relying on their values."""

import pytest

import tilelang
from test_pto_parallel_gather_scatter import capture_passes, calls, dump_evidence
from test_pto_parallel_indexed_uniform import uniform_store_kernel


@pytest.mark.pto
@pytest.mark.parametrize(
    "value_kind,extent,select_target",
    [("scalar", 256, False), ("vector", 256, False), ("scalar", 150, False), ("vector", 150, False), ("vector", 256, True)],
)
def test_uniform_scatter_compiles_with_duplicate_active_offsets(value_kind, extent, select_target):
    name = f"uniform_scatter_compile_{value_kind}_{extent}_{select_target}"
    cache_enabled = tilelang.is_cache_enabled()
    tilelang.disable_cache()
    try:
        with capture_passes() as captured:
            kernel = tilelang.compile(uniform_store_kernel(value_kind, extent, select_target=select_target), target="pto", out_idx=None)
    finally:
        if cache_enabled:
            tilelang.enable_cache()
    source = kernel.get_kernel_source()
    dump_evidence(name, captured, source)
    after = captured["VectorizeParallelToPTO_after"]
    assert len(calls(after, "tl.vmi.vscatter")) == 1
    assert not calls(after, "tl.vmi.vgather")
    assert "pto.vmi.vscatter(" in source
    # Deliberately do not execute: the written value for duplicate active
    # offsets is undefined, including when every lane supplies the same value.
