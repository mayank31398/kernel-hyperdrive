import torch
import triton

from ...enums import KernelBackend
from ...utils import CutoTuneParameter, get_sm_count
from .triton_implementation import contiguous_count_triton_kernel


def contiguous_count_khd(
    x: torch.Tensor,
    start: int,
    end: int,
    kernel_backend: KernelBackend | CutoTuneParameter = CutoTuneParameter(),
    BLOCK_SIZE: int | CutoTuneParameter = CutoTuneParameter(),
) -> torch.Tensor:
    B = get_sm_count(x.device)
    output = torch.empty(B, end - start, dtype=torch.long, device=x.device)

    num_elements = x.numel()

    if kernel_backend == KernelBackend.triton:
        contiguous_count_triton_kernel[(B,)](
            x_ptr=x,
            output_ptr=output,
            B=B,
            output_stride_b=output.stride(0),
            num_elements=num_elements,
            start=start,
            end=end,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return output.sum(dim=0)
