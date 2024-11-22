import torch
import triton

from ...enums import KernelBackend
from ...utils import CutoTuneParameter, ceil_divide, ensure_contiguous, get_sm_count
from .naive_implementation import contiguous_count_naive_kernel
from .triton_implementation import contiguous_count_triton_kernel


@ensure_contiguous
def contiguous_count_khd(
    x: torch.Tensor,
    start: int,
    end: int,
    kernel_backend: KernelBackend | CutoTuneParameter = CutoTuneParameter(),
    BLOCK_SIZE_B: int | CutoTuneParameter = CutoTuneParameter(),
) -> torch.Tensor:
    B = x.numel()
    C = end - start
    BLOCK_SIZE_C = triton.next_power_of_2(C)

    sm_count = get_sm_count(x.device)
    num_programs = min(sm_count, ceil_divide(B, BLOCK_SIZE_B))

    output = torch.zeros(num_programs, C, dtype=torch.long, device=x.device)

    if kernel_backend == KernelBackend.triton:
        contiguous_count_triton_kernel[(num_programs,)](
            x_ptr=x,
            output_ptr=output,
            output_stride_b=output.stride(0),
            B=B,
            C=C,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_C=BLOCK_SIZE_C,
        )
    elif kernel_backend == KernelBackend.naive:
        contiguous_count_naive_kernel(
            x=x.view(-1),
            output=output,
            num_programs=num_programs,
            B=B,
            C=C,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_C=BLOCK_SIZE_C,
        )

    return output.sum(dim=0)
