import torch

from .....constants import LIBRARY_NAME
from .....kernel_registry import KernelRegistry


_KERNEL_NAME = "add_tensor_forward_cuda"


def _add_tensor_forward_cuda_kernel(
    x: torch.Tensor, y: torch.Tensor, output: torch.Tensor, vector_instruction_width: int, BLOCK_SIZE: int
) -> None:
    KernelRegistry.get_kernel(_KERNEL_NAME)(x, y, output, vector_instruction_width, BLOCK_SIZE)


@torch.library.custom_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"output"})
def _add_tensor_forward_cuda_kernel_compileable(
    x: torch.Tensor, y: torch.Tensor, output: torch.Tensor, vector_instruction_width: int, BLOCK_SIZE: int
) -> None:
    _add_tensor_forward_cuda_kernel(
        x=x, y=y, output=output, vector_instruction_width=vector_instruction_width, BLOCK_SIZE=BLOCK_SIZE
    )


def add_tensor_forward_cuda(
    x: torch.Tensor, y: torch.Tensor, output: torch.Tensor, vector_instruction_width: int, BLOCK_SIZE: int
) -> None:
    function = (
        _add_tensor_forward_cuda_kernel_compileable
        if torch.compiler.is_compiling()
        else _add_tensor_forward_cuda_kernel
    )
    return function(x=x, y=y, output=output, vector_instruction_width=vector_instruction_width, BLOCK_SIZE=BLOCK_SIZE)
