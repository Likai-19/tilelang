"""NPU numerics for lane-dependent metadata in real TileLang view kernels."""

import pytest
import torch

from test_pto_parallel_gather_scatter_run import _run, npu as _npu
from test_pto_parallel_indexed_metadata import continuous_payload_view_stride, payload_view_stride_kernel, unused_loaded_stride_kernel

npu = _npu


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
def test_payload_view_stride_numerics(kind, npu):
    source = torch.arange(512 if kind == "gather" else 128, dtype=torch.float32) % 113 - 56
    index = torch.arange(128, dtype=torch.int32)
    output = torch.full((128 if kind == "gather" else 512,), 211.0)
    offsets = torch.arange(128) * 3
    expected = output.clone()
    if kind == "gather":
        expected[:] = source[offsets]
    else:
        expected[offsets] = source
    _run(f"payload_stride_{kind}", payload_view_stride_kernel(kind), [source, index, output], expected, npu)


@pytest.mark.pto
@pytest.mark.parametrize("kind", ["gather", "scatter"])
def test_b16_large_owner_coordinate_view_stride_numerics(kind, npu):
    source = (torch.arange(70000 if kind == "gather" else 128, dtype=torch.float32) % 113 - 56).to(torch.float16)
    index = torch.zeros((128,), dtype=torch.uint16)
    output = torch.full((128 if kind == "gather" else 70000,), 211, dtype=torch.float16)
    offsets = torch.arange(128) * 2
    expected = output.clone()
    if kind == "gather":
        expected[:] = source[offsets]
    else:
        expected[offsets] = source
    _run(
        f"b16_coordinate_payload_stride_{kind}",
        payload_view_stride_kernel(kind, "float16", 70000, loaded_index=False),
        [source, index, output],
        expected,
        npu,
    )


@pytest.mark.pto
def test_continuous_payload_view_stride_numerics(npu):
    source = torch.arange(512, dtype=torch.float32) % 113 - 56
    output = torch.full((256,), 211.0)
    _run("continuous_payload_stride", continuous_payload_view_stride, [source, output], source[:256], npu)


@pytest.mark.pto
@pytest.mark.parametrize("uniform", [False, True])
def test_unused_loaded_view_stride_numerics(uniform, npu):
    source = torch.arange(256, dtype=torch.float32) % 113 - 56
    index = torch.arange(256, dtype=torch.int32) + 7
    output = torch.full((256,), 211.0)
    expected = source[:1].expand(256).clone() if uniform else source.clone()
    _run(f"unused_loaded_stride_{uniform}", unused_loaded_stride_kernel(uniform), [source, index, output], expected, npu)
