import torch

from .....constants import LIBRARY_NAME
from .....kernel_registry import KernelRegistry
from ....utils import torch_custom_op


_KERNEL_NAME = "add_scalar_forward_cuda"


def _add_scalar_forward_cuda(x: torch.Tensor, y: float, output: torch.Tensor, BLOCK_SIZE: int) -> None:
    KernelRegistry.get_kernel(_KERNEL_NAME)(x, y, output, BLOCK_SIZE)


@torch_custom_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"output"})
def _add_scalar_forward_cuda_compilable(x: torch.Tensor, y: float, output: torch.Tensor, BLOCK_SIZE: int) -> None:
    _add_scalar_forward_cuda(x, y, output, BLOCK_SIZE)
