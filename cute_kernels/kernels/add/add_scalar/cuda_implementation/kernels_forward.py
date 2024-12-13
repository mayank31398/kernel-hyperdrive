import torch

from .....constants import LIBRARY_NAME
from .....kernel_registry import KernelRegistry
from .....utils import cute_op


_KERNEL_NAME = "add_scalar_forward_cuda"


@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"output"})
def add_scalar_forward_cuda(
    x: torch.Tensor, y: float, output: torch.Tensor, vector_instruction_width: int, BLOCK_SIZE: int
) -> None:
    KernelRegistry.get_kernel(_KERNEL_NAME)(x, y, output, vector_instruction_width, BLOCK_SIZE)
