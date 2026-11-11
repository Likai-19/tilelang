"""NPU numerics for B16 offsets generated in their proven uint16 range."""

import pytest
import torch

from test_pto_parallel_gather_scatter_run import _run, npu as _npu
from test_pto_parallel_indexed_range import affine_b16_scatter_kernel, b16_transposed_index_kernel, block_coordinate_b16_scatter

npu = _npu


@pytest.mark.pto
@pytest.mark.parametrize("force_scatter", [False, True])
def test_b16_transposed_index_numerics(force_scatter, npu):
    source = (torch.arange(256, dtype=torch.float32) % 113 - 56).to(torch.float16)
    index = (torch.arange(128, dtype=torch.int64) * 37 + 11) % 256
    gathered = source[index].expand(128, 128).contiguous()
    if force_scatter:
        expected = torch.stack([gathered, source[:128].expand(128, 128)])
    else:
        expected = gathered
    output = torch.full_like(expected, 211)
    _run(
        f"b16_transposed_index_range_{force_scatter}",
        b16_transposed_index_kernel(force_scatter),
        [source, index.to(torch.uint16), output],
        expected,
        npu,
    )


@pytest.mark.pto
@pytest.mark.parametrize(
    "extent,stride,origin,output_size",
    [(150, 2, 1, 512), (128, 512, 511, 65536), (127, 516, 519, 65536), (255, 258, 3, 65536), (128, 2, 0, 70000)],
)
def test_affine_b16_scatter_numerics(extent, stride, origin, output_size, npu):
    source = (torch.arange(256, dtype=torch.float32) % 113 - 56).to(torch.float16)
    output = torch.full((output_size,), 211, dtype=torch.float16)
    expected = output.clone()
    expected[torch.arange(extent) * stride + origin] = source[:extent]
    _run(
        f"affine_b16_range_{extent}_{stride}_{origin}",
        affine_b16_scatter_kernel(extent, stride, origin, output_size),
        [source, output],
        expected,
        npu,
    )


@pytest.mark.pto
def test_block_coordinate_b16_scatter_numerics(npu):
    source = (torch.arange(256, dtype=torch.float32) % 113 - 56).to(torch.float16)
    output = torch.full((2, 512), 211, dtype=torch.float16)
    expected = output.clone()
    expected[0, torch.arange(128) * 2] = source[:128]
    expected[1, torch.arange(128) * 2 + 1] = source[:128]
    _run("block_coordinate_b16_scatter", block_coordinate_b16_scatter, [source, output], expected, npu)
