"""NPU acceptance for a lexical owner inside SIMD_VF."""

import pytest
import torch
import torch_npu

from test_pto_parallel_gather_scatter_run import _run
from test_pto_parallel_indexed_scopes import nested_scatter_then_read


@pytest.mark.pto
def test_nested_scatter_owner_is_read_after_its_parallel():
    if not torch.npu.is_available():
        pytest.skip("NPU runtime unavailable")
    torch_npu.npu.set_compile_mode(jit_compile=False)
    device = torch.device("npu")
    values = torch.arange(256, dtype=torch.float32)
    index = (torch.arange(256, dtype=torch.int64) * 37 + 11) % 256
    expected = torch.empty_like(values)
    expected[index] = values
    _run("nested_scatter_then_read", nested_scatter_then_read, [values, index.to(torch.int32), torch.zeros_like(values)], expected, device)
