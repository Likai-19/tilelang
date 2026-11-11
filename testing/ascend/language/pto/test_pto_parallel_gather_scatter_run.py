"""True NPU acceptance for automatic Parallel gather/scatter.

Inputs and every untouched output element are initialized on CPU. All kernels
are compiled from scalar T.Parallel programs; no hand-written VMI substitution.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

import tilelang

_spec = importlib.util.spec_from_file_location("pto_parallel_gs_contract", Path(__file__).with_name("test_pto_parallel_gather_scatter.py"))
gs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gs)


@pytest.fixture(scope="module")
def npu():
    if not torch.npu.is_available():
        pytest.skip("NPU runtime unavailable")
    torch_npu.npu.set_compile_mode(jit_compile=False)
    return torch.device("npu")


def _payload(size, dtype=torch.float32):
    values = torch.arange(size, dtype=torch.int64) % 113 - 56
    if dtype == torch.uint32:
        values += 0x80000000
    return values.to(dtype)


def _permutation(size):
    return (torch.arange(size, dtype=torch.int64) * 37 + 11) % size


def _record(name, result):
    root = os.environ.get("TILELANG_GS_EVIDENCE_DIR")
    if root is None:
        return
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    result = {"name": name, **result}
    with (path / "runtime_results.jsonl").open("a") as file:
        file.write(json.dumps(result) + "\n")


def _run(name, func, args, expected, device, *, auto_schedule=True):
    captured = {}
    source = None
    try:
        cache_enabled = tilelang.is_cache_enabled()
        tilelang.disable_cache()
        try:
            with gs.capture_passes() as captured:
                kernel = tilelang.compile(
                    func,
                    target="pto",
                    out_idx=None,
                    pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: auto_schedule},
                )
        finally:
            if cache_enabled:
                tilelang.enable_cache()
        source = kernel.get_kernel_source()
        assert "pto.set_flag(" in source and "pto.wait_flag(" in source
        gs.dump_evidence(name, captured, source)
        device_args = [value.to(device) for value in args]
        kernel(*device_args)
        torch.npu.synchronize()
        actual = device_args[-1].cpu()
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected), (
            f"{name}: wrong output at {torch.nonzero(actual != expected).tolist()[:10]}\n"
            f"actual={actual.flatten().tolist()[:16]}\nexpected={expected.flatten().tolist()[:16]}"
        )
        _record(name, {"pass": True, "auto_schedule": auto_schedule, "device": str(device)})
    except Exception as error:
        gs.dump_evidence(name, captured, source)
        _record(name, {"pass": False, "auto_schedule": auto_schedule, "error": str(error)})
        raise


@pytest.mark.pto
@pytest.mark.parametrize("name", gs.DESIGN_CASES)
def test_design_kernel_numerics(name, npu):
    if name in ("G1_gather_f32", "S1_scatter_f32"):
        x = _payload(256)
        idx = _permutation(256)
        out = torch.full((256,), 211.0)
        expected = out.clone()
        if name.startswith("G"):
            expected[:] = x[idx]
        else:
            expected[idx] = x
        args = [x, idx.to(torch.int32), out]
    elif name == "G2_gather_f16_tail":
        x = _payload(256, torch.float16)
        idx = _permutation(256)
        idx[150:] = -1
        out = torch.full((256,), 211, dtype=torch.float16)
        expected = out.clone()
        expected[:150] = x[idx[:150]]
        args = [x, idx.to(torch.int16), out]
    elif name == "S3_stride2":
        x = _payload(256)
        out = torch.full((512,), 211.0)
        expected = out.clone()
        expected[torch.arange(256) * 2 + 1] = x
        args = [x, out]
    elif name == "M1_saved_index":
        x = _payload(256)
        idx = _permutation(256).to(torch.int32)
        out = torch.full((256,), 211.0)
        expected = x[idx.to(torch.int64)].clone()
        args = [x, idx, out]
    elif name in ("U2_dynamic_row", "D2_column_gather", "V2_versioned"):
        x = _payload(1024).reshape(4, 256)
        out = torch.full((4, 256), 211.0)
        if name == "U2_dynamic_row":
            idx = torch.tensor([3, 1, 0, 2], dtype=torch.int32)
            expected = x[idx.to(torch.int64), :].clone()
        elif name == "D2_column_gather":
            idx = _permutation(256).to(torch.int32)
            expected = x[:, idx.to(torch.int64)].clone()
        else:
            idx = torch.stack([(_permutation(256) + row * 17) % 256 for row in range(4)]).to(torch.int32)
            expected = torch.stack([x[row, idx[row].to(torch.int64)] for row in range(4)])
        args = [x, idx, out]
    elif name == "V1_strided_view":
        x = _payload(800)
        idx = _permutation(256).to(torch.int32)
        out = torch.full((256,), 211.0)
        expected = x[544 + idx.to(torch.int64)].clone()
        args = [x, idx, out]
    elif name == "G3_nested_gather":
        x = _payload(256)
        idx0 = _permutation(256).to(torch.int32)
        idx1 = _permutation(256).to(torch.int32)
        out = torch.full((256,), 211.0)
        expected = x[idx1[idx0.to(torch.int64)].to(torch.int64)].clone()
        args = [x, idx0, idx1, out]
    elif name == "O1_alias_dependency":
        x = _payload(256)
        idx = _permutation(256).to(torch.int32)
        out = torch.full((256,), 211.0)
        expected = x[idx.to(torch.int64)].clone()
        args = [x, idx, out]
    else:
        raise AssertionError(name)
    _run(name, gs.DESIGN_CASES[name], args, expected, npu)


@pytest.mark.pto
def test_transpose_with_continuous_load_and_indexed_store(npu):
    source = _payload(1024).reshape(4, 256)
    output = torch.full((256, 4), 211.0)
    _run("transpose_2d", gs.transpose_2d, [source, output], source.T.contiguous(), npu)


@pytest.mark.pto
def test_2d_candidate_counts_nested_index_loads_numerics(npu):
    source = _payload(16384)
    index = _permutation(16384).reshape(128, 128).to(torch.int32)
    output = torch.full((128, 128), 211.0)
    expected = source[index.T.to(torch.int64)].contiguous()
    _run("candidate_nested_index", gs.nested_index_candidate, [source, index, output], expected, npu)


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["load_count", "store_count", "equal_counts", "scatter_count"])
def test_2d_candidate_ranking_numerics(kind, npu):
    source = _payload(128 * 128).reshape(128, 128)
    output = torch.full((2, 128, 128), 211.0)
    expected = output.clone()
    expected[0] = source.T if kind == "scatter_count" else source + source.T
    if kind == "load_count":
        expected[1] = source.T
    elif kind != "scatter_count":
        expected[1] = 1.0
    _run(f"candidate_{kind}", gs.candidate_ranking_kernel(kind), [source, output], expected, npu)


@pytest.mark.pto
@pytest.mark.parametrize(
    "kind,dtype,lanes",
    [
        ("gather", "float32", 64),
        ("gather", "float32", 256),
        ("gather", "float16", 64),
        ("gather", "bfloat16", 256),
        ("scatter", "float32", 64),
        ("scatter", "float16", 64),
        ("scatter", "bfloat16", 256),
        ("gather", "int32", 128),
        ("scatter", "int16", 128),
        ("gather", "uint32", 128),
    ],
)
def test_dtype_width_tail_and_canary(kind, dtype, lanes, npu):
    extent = lanes + 2
    padded = 2 * lanes
    owner = max(256, padded)
    torch_dtype = getattr(torch, dtype)
    x = _payload(owner, torch_dtype)
    idx = _permutation(padded)
    if kind == "gather":
        idx[extent:] = -1
    out = torch.full((owner,), 211, dtype=torch_dtype)
    expected = out.to(torch.int64)
    if kind == "gather":
        expected[:extent] = x.to(torch.int64)[idx[:extent]]
    else:
        expected[idx[:extent]] = x.to(torch.int64)[:extent]
    _run(
        f"{kind}_{dtype}_{lanes}_tail",
        gs.indexed_kernel(kind, dtype, lanes, extent),
        [x, idx.to(torch.int16 if dtype in ("float16", "bfloat16", "int16", "uint16") else torch.int32), out],
        expected.to(torch_dtype),
        npu,
    )


@pytest.mark.pto
def test_repeated_gather_index_is_not_a_scatter_conflict(npu):
    x = _payload(256)
    idx = (torch.arange(256) % 7).to(torch.int32)
    out = torch.full((256,), 211.0)
    _run("gather_repeat", gs.indexed_kernel(), [x, idx, out], x[idx.to(torch.int64)], npu)


@pytest.mark.pto
def test_manual_outer_pipe_protocol_without_autoschedule(npu):
    x = _payload(256)
    idx = _permutation(256).to(torch.int32)
    out = torch.full((256,), 211.0)
    _run(
        "gather_manual_schedule",
        gs.indexed_kernel(manual_sync=True),
        [x, idx, out],
        x[idx.to(torch.int64)],
        npu,
        auto_schedule=False,
    )


@pytest.mark.pto
def test_sequential_gather_and_elementwise_units_keep_their_own_masks(npu):
    x = _payload(256)
    index = _permutation(256)
    output = torch.full((256,), 211.0)
    expected = output.clone()
    expected[:150] = x[index[:150]] + x[:150]
    expected += x[index]
    expected[:150] += x[index[:150]]
    _run("mixed_indexed_units", gs.mixed_indexed_units_kernel(), [x, index.to(torch.int32), output], expected, npu)


@pytest.mark.pto
def test_loaded_offset_plus_coordinate_uses_old_varying_value(npu):
    x = _payload(256)
    idx = _permutation(128).to(torch.int32)
    out = torch.full((128,), 211.0)
    expected = x[idx.to(torch.int64) + torch.arange(128)]
    _run("saved_varying_plus_coordinate", gs.saved_offset_kernel(), [x, idx, out], expected, npu)


@pytest.mark.pto
def test_b16_large_owner_small_window(npu):
    x = _payload(70000, torch.float16)
    idx = _permutation(128)
    output = torch.full((128,), 211, dtype=torch.float16)
    expected = x[66000 + idx]
    _run("b16_rebased_window", gs.b16_rebased_window_kernel(), [x, idx.to(torch.uint16), output], expected, npu)


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
@pytest.mark.parametrize("extent", [22, 278])
def test_unified_loop_tail_only_and_multiple_full_chunks(kind, extent, npu):
    lanes = 128
    padded = (extent + lanes - 1) // lanes * lanes
    owner = max(256, padded)
    x = _payload(owner)
    index = _permutation(padded)
    if kind == "gather":
        index[extent:] = -1
    output = torch.full((owner,), 211.0)
    expected = output.clone()
    if kind == "gather":
        expected[:extent] = x[index[:extent]]
    else:
        expected[index[:extent]] = x[:extent]
    _run(f"unified_{kind}_{extent}", gs.indexed_kernel(kind, lanes=lanes, extent=extent), [x, index.to(torch.int32), output], expected, npu)


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
def test_unified_2d_rows_keep_independent_tail_addresses(kind, npu):
    x = _payload(3 * 256).reshape(3, 256)
    index = torch.stack([(_permutation(256) + row * 13) % 256 for row in range(3)])
    if kind == "gather":
        index[:, 150:] = -1
    output = torch.full((3, 256), 211.0)
    expected = output.clone()
    for row in range(3):
        if kind == "gather":
            expected[row, :150] = x[row, index[row, :150]]
        else:
            expected[row, index[row, :150]] = x[row, :150]
    _run(f"unified_2d_{kind}", gs.indexed_2d_tail_kernel(kind), [x, index.to(torch.int32), output], expected, npu)


@pytest.mark.pto
def test_unified_loop_keeps_original_loaded_index_after_mutation(npu):
    x = _payload(384)
    index = _permutation(384)
    index[278:] = -1
    output = torch.full((384,), 211.0)
    expected = output.clone()
    expected[:278] = x[index[:278]]
    _run("unified_saved_index", gs.saved_index_loop_kernel(), [x, index.to(torch.int32), output], expected, npu)


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
@pytest.mark.parametrize(
    "index_dtype,payload_dtype",
    [("int8", "float32"), ("int16", "bfloat16"), ("int32", "float32"), ("int16", "float16"), ("int8", "float16")],
)
def test_signed_dsl_offsets_preserve_high_bit_indices(kind, index_dtype, payload_dtype, npu):
    dtype = getattr(torch, payload_dtype)
    index = (torch.arange(128, dtype=torch.int64) * 53 + 129) % 256
    x = _payload(256, dtype)
    output = torch.full((256,), 211, dtype=dtype)
    expected = output.clone()
    if kind == "gather":
        expected[:128] = x[index]
    else:
        expected[index] = x[:128]
    _run(
        f"dsl_{kind}_{index_dtype}_{payload_dtype}",
        gs.handwritten_signed_offset_kernel(kind, index_dtype, payload_dtype),
        [x, index.to(getattr(torch, index_dtype)), output],
        expected,
        npu,
    )


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
def test_signed_int16_dsl_offsets_preserve_high_bit(npu, kind):
    index = 32768 + (torch.arange(128, dtype=torch.int64) * 53) % 512
    x = _payload(40000, torch.float16)
    output = torch.full((40000,), 211, dtype=torch.float16)
    expected = output.clone()
    if kind == "gather":
        expected[:128] = x[index]
    else:
        expected[index] = x[:128]
    _run(
        f"dsl_{kind}_int16_highbit",
        gs.handwritten_signed_offset_kernel(kind, "int16", "float16", 40000),
        [x, index.to(torch.int16), output],
        expected,
        npu,
    )


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
@pytest.mark.parametrize("index_dtype", ["int16", "uint16"])
@pytest.mark.parametrize("extent", [256, 150])
def test_b32_native16_offsets_are_widened_by_dsl(kind, index_dtype, extent, npu):
    x = _payload(256)
    index = _permutation(256)
    output = torch.full((256,), 211.0)
    expected = output.clone()
    if kind == "gather":
        expected[:extent] = x[index[:extent]]
    else:
        expected[index[:extent]] = x[:extent]
    index[extent:] = -1 if index_dtype == "int16" else 65535
    _run(
        f"b32_native16_{kind}_{index_dtype}_{extent}",
        gs.indexed_kernel(kind, "float32", 128, extent, index_dtype=index_dtype),
        [x, index.to(getattr(torch, index_dtype)), output],
        expected,
        npu,
    )


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
@pytest.mark.parametrize("index_dtype", ["int16", "uint16"])
def test_b32_dsl_native16_offsets_preserve_unsigned_high_bit(kind, index_dtype, npu):
    index = 32768 + (torch.arange(128, dtype=torch.int64) * 53) % 512
    x = _payload(40000 if kind == "gather" else 128)
    output = torch.full((128 if kind == "gather" else 40000,), 211.0)
    expected = output.clone()
    if kind == "gather":
        expected[:128] = x[index]
    else:
        expected[index] = x[:128]
    _run(
        f"b32_dsl_{kind}_{index_dtype}_highbit",
        gs.handwritten_signed_offset_kernel(kind, index_dtype, "float32", 40000, compact=True),
        [x, index.to(getattr(torch, index_dtype)), output],
        expected,
        npu,
    )


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
@pytest.mark.parametrize("index_dtype", ["int16", "uint16"])
@pytest.mark.parametrize("extent", [256, 278])
def test_native_b16_indices_need_no_width_round_trip(kind, index_dtype, extent, npu):
    owner = (extent + 127) // 128 * 128
    x = _payload(owner, torch.float16)
    index = _permutation(owner)
    output = torch.full((owner,), 211, dtype=torch.float16)
    expected = output.clone()
    if kind == "gather":
        expected[:extent] = x[index[:extent]]
    else:
        expected[index[:extent]] = x[:extent]
    index[extent:] = -1 if index_dtype == "int16" else 65535
    _run(
        f"native16_{kind}_{index_dtype}_{extent}",
        gs.indexed_kernel(kind, "float16", 128, extent, index_dtype=index_dtype),
        [x, index.to(getattr(torch, index_dtype)), output],
        expected,
        npu,
    )


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
def test_native_uint16_indices_keep_high_bit_positions(kind, npu):
    x = _payload(40000, torch.float16)
    index = 32768 + (torch.arange(384, dtype=torch.int64) * 53) % 1024
    output = torch.full((40000,), 211, dtype=torch.float16)
    expected = output.clone()
    if kind == "gather":
        expected[:278] = x[index[:278]]
    else:
        expected[index[:278]] = x[:278]
    index[278:] = 65535

    @gs.T.prim_func
    def main(A: gs.T.Tensor((40000,), "float16"), I: gs.T.Tensor((384,), "uint16"), C: gs.T.Tensor((40000,), "float16")):
        with gs.T.Kernel(1):
            x_ub = gs.T.alloc_shared((40000,), "float16")
            idx_ub = gs.T.alloc_shared((384,), "uint16")
            y_ub = gs.T.alloc_shared((40000,), "float16")
            gs.T.copy(A, x_ub)
            gs.T.copy(I, idx_ub)
            gs.T.copy(C, y_ub)
            with gs.T.SimdVF(lanes=128):
                for i in gs.T.Parallel(278):
                    if kind == "gather":
                        y_ub[i] = x_ub[idx_ub[i]]
                    else:
                        y_ub[idx_ub[i]] = x_ub[i]
            gs.T.copy(y_ub, C)

    _run(f"native16_{kind}_uint16_highbit", main, [x, index.to(torch.uint16), output], expected, npu)


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
def test_native16_nonzero_origin_preserves_wide_math(kind, npu):
    x = _payload(40400 if kind == "gather" else 384, torch.float32)
    index = _permutation(384)
    output = torch.full((384 if kind == "gather" else 40400,), 211, dtype=torch.float32)
    expected = output.clone()
    if kind == "gather":
        expected[:278] = x[40000 + index[:278]]
    else:
        expected[40000 + index[:278]] = x[:278]
    index[278:] = 65535
    _run(
        f"native16_{kind}_offset_view",
        gs.native16_offset_view_kernel(kind, "float32"),
        [x, index.to(torch.uint16), output],
        expected,
        npu,
    )


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
def test_b16_explicit_source_index_narrowing(kind, npu):
    x = _payload(256, torch.float16)
    index = _permutation(256)
    output = torch.full((256,), 211, dtype=torch.float16)
    expected = output.clone()
    if kind == "gather":
        expected[:150] = x[index[:150]]
    else:
        expected[index[:150]] = x[:150]
    index[150:] = -1
    _run(f"explicit_narrow_{kind}", gs.explicit_narrow_index_kernel(kind), [x, index.to(torch.int32), output], expected, npu)
