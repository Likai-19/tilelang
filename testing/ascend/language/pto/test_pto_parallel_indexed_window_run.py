"""NPU verification of block-dependent windows in a large B16 owner."""

import pytest
import torch

from test_pto_parallel_indexed_window import outer_range_window_kernel
from test_pto_parallel_gather_scatter_run import _run


@pytest.mark.pto
def test_b16_window_with_block_dependent_base_numerics():
    import torch_npu

    if not torch.npu.is_available():
        pytest.skip("NPU runtime unavailable")
    torch_npu.npu.set_compile_mode(jit_compile=False)
    device = torch.device("npu")
    source = (torch.arange(70000, dtype=torch.int64) % 113 - 56).to(torch.float16)
    index = (torch.arange(128, dtype=torch.int64) * 37 + 11) % 256
    output = torch.full((2, 128), 211, dtype=torch.float16)
    expected = torch.stack([source[index], source[35000 + index]])
    _run("outer_range_window", outer_range_window_kernel, [source, index.to(torch.uint16), output], expected, device)
