import torch

from .....constants import LIBRARY_NAME
from .....kernel_registry import KernelRegistry
from ....utils import torch_custom_op


_KERNEL_NAME = "add_tensor_forward_cuda"


def _add_tensor_forward_cuda(
    x: torch.Tensor, y: torch.Tensor, vectorized_loop_size: int, BLOCK_SIZE: int
) -> torch.Tensor:
    return KernelRegistry.get_kernel(_KERNEL_NAME)(x, y, vectorized_loop_size, BLOCK_SIZE)


@torch_custom_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={})
def _add_tensor_forward_cuda_compilable(
    x: torch.Tensor, y: torch.Tensor, vectorized_loop_size: int, BLOCK_SIZE: int
) -> torch.Tensor:
    return _add_tensor_forward_cuda(x=x, y=y, vectorized_loop_size=vectorized_loop_size, BLOCK_SIZE=BLOCK_SIZE)


@_add_tensor_forward_cuda_compilable.register_fake
def _(x: torch.Tensor, y: torch.Tensor, vectorized_loop_size: int, BLOCK_SIZE: int) -> torch.Tensor:
    return torch.empty_like(x)
