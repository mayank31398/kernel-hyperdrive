import torch

from .....constants import LIBRARY_NAME
from .....kernel_registry import KernelRegistry
from ....utils import torch_custom_op


_KERNEL_NAME = "add_scalar_forward_cuda"


@torch_custom_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"output"})
def _(x: torch.Tensor, y: float, output: torch.Tensor, BLOCK_SIZE: int) -> None:
    KernelRegistry.get_kernel(_KERNEL_NAME)(x, y, output, BLOCK_SIZE)


class _AddScalar_CUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: float, BLOCK_SIZE: int) -> torch.Tensor:
        if y == 0:
            return x

        output = torch.empty_like(x)

        torch.ops.khd.add_scalar_forward_cuda(x=x, y=y, output=output, BLOCK_SIZE=BLOCK_SIZE)

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        return output_grad, None, None


def add_scalar_cuda(x: torch.Tensor, y: float, BLOCK_SIZE: int) -> torch.Tensor:
    """tensor addition

    Args:
        x (torch.Tensor): input tensor
        y (float): input scalar
        BLOCK_SIZE (int): block size

    Returns:
        torch.Tensor: output tensor
    """

    return _AddScalar_CUDA.apply(x, y, BLOCK_SIZE)
