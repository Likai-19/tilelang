"""NPU verification of scalar coordinates inside full indexed offsets."""

import pytest
import torch

from test_pto_parallel_gather_scatter_run import _run, npu as _npu
from test_pto_parallel_indexed_offset_expr import (
    nested_b16_rebased_gather_kernel,
    scalar_floor_coordinate_kernel,
    scalar_select_coordinate_cast_kernel,
)

npu = _npu


@pytest.mark.pto
@pytest.mark.parametrize("extent", [256, 150])
def test_scalar_select_cast_keeps_indexed_address_semantics(extent, npu):
    source = torch.arange(512, dtype=torch.float32) - 123
    index = (torch.arange(256, dtype=torch.int64) * 37 + 11) % 128
    index[0] = 0
    output = torch.full((256,), -777.0)
    expected = output.clone()
    offset = index[:extent] + 1
    expected[:extent] = source[offset]
    index[extent:] = -1
    _run(
        f"scalar_select_coordinate_cast_{extent}",
        scalar_select_coordinate_cast_kernel(extent),
        [source, index.to(torch.int32), output],
        expected,
        npu,
    )


@pytest.mark.pto
def test_nested_b16_rebased_gather_numerics(npu):
    source = torch.arange(70000, dtype=torch.int64) % 256
    index = (torch.arange(128, dtype=torch.int64) * 37 + 11) % 256
    expected = index.clone()
    for _ in range(14):
        expected = source[66000 + expected]
    output = torch.full((128,), 211, dtype=torch.uint16)
    _run(
        "nested_b16_rebased_gather",
        nested_b16_rebased_gather_kernel,
        [source.to(torch.uint16), index.to(torch.uint16), output],
        expected.to(torch.uint16),
        npu,
    )


@pytest.mark.pto
def test_scalar_floor_coordinate_gather_numerics(npu):
    # `bx % 2` stays one scalar coordinate per launch block and adds to the
    # loaded offsets; every block must gather with its own broadcast value.
    source = torch.arange(256, dtype=torch.float32) - 123
    index = (torch.arange(128, dtype=torch.int64) * 37 + 11) % 200
    output = torch.full((4, 256), -777.0)
    expected = output.clone()
    for bx in range(4):
        expected[bx, :128] = source[index + bx % 2]
    _run(
        "scalar_floor_coordinate_gather",
        scalar_floor_coordinate_kernel("gather"),
        [source, index.to(torch.int32), output],
        expected,
        npu,
    )
