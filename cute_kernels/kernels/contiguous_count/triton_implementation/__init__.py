import torch
import triton

from ....constants import LIBRARY_NAME
from ....utils import ceil_divide, cute_op, get_next_power_of_2, get_sm_count
from .kernels_forward import contiguous_count_triton_kernel


_KERNEL_NAME = "contiguous_count_triton"


def _fake(x: torch.Tensor, size: int, BLOCK_SIZE_B: int) -> torch.Tensor:
    return torch.empty(size, dtype=torch.long, device=x.device)


@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={}, fake_func=_fake)
def contiguous_count_triton(x: torch.Tensor, size: int, BLOCK_SIZE_B: int) -> torch.Tensor:
    B = x.numel()
    BLOCK_SIZE_C = get_next_power_of_2(size)

    sm_count = get_sm_count(x.device)
    num_programs = min(sm_count, ceil_divide(B, BLOCK_SIZE_B))

    output = torch.zeros(num_programs, size, dtype=torch.long, device=x.device)

    with torch.device(x.device):
        contiguous_count_triton_kernel[(num_programs,)](
            x_ptr=x,
            output_ptr=output,
            B=B,
            C=size,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_C=BLOCK_SIZE_C,
        )

    return output.sum(dim=0)
