import torch

from .....constants import LIBRARY_NAME
from .....kernel_registry import KernelRegistry


_KERNEL_NAME = "add_scalar_forward_cuda"


def add_scalar_forward_cuda_kernel(x: torch.Tensor, y: float, output: torch.Tensor, BLOCK_SIZE: int) -> None:
    KernelRegistry.get_kernel(_KERNEL_NAME)(x, y, output, BLOCK_SIZE)


@torch.library.custom_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"output"})
def add_scalar_forward_cuda_kernel_compileable(
    x: torch.Tensor, y: float, output: torch.Tensor, BLOCK_SIZE: int
) -> None:
    add_scalar_forward_cuda_kernel(x=x, y=y, output=output, BLOCK_SIZE=BLOCK_SIZE)
