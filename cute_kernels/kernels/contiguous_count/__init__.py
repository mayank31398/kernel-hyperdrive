import torch

from ...enums import KernelBackend
from ...utils import ensure_contiguous
from .triton_implementation import contiguous_count_triton


@torch.no_grad()
@ensure_contiguous
def contiguous_count_cute(
    x: torch.Tensor,
    size: int,
    kernel_backend: KernelBackend = KernelBackend.triton,
    BLOCK_SIZE_B: int = 64,
) -> torch.Tensor:
    assert x.dim() == 1, "x should be 1-dimensional"
    assert x.dtype in [torch.int32, torch.long]

    if kernel_backend == KernelBackend.triton:
        output = contiguous_count_triton(x=x, size=size, BLOCK_SIZE_B=BLOCK_SIZE_B)
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output
