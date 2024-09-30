import torch

from .....constants import LIBRARY_NAME, TORCH_VER_SUPPORTS_COMPILE
from .....kernel_registry import KernelRegistry
from .....utils import requires_package


_KERNEL_NAME = "add_scalar_forward_cuda"


if requires_package("torch", TORCH_VER_SUPPORTS_COMPILE):
    @torch.library.custom_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={})
    def _add_tensor_forward_cuda_compilable(x: torch.Tensor, y: float, BLOCK_SIZE: int) -> torch.Tensor:
        return KernelRegistry.get_kernel(_KERNEL_NAME)(x, y, BLOCK_SIZE)


    @_add_tensor_forward_cuda_compilable.register_fake
    def _(x: torch.Tensor, y: float, BLOCK_SIZE: int) -> torch.Tensor:
        return torch.empty_like(x)


class _AddScalar_CUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: float) -> torch.Tensor:
        if y == 0:
            return x

        BLOCK_SIZE = 1024

        if torch.compiler.is_compiling():
            output = _add_tensor_forward_cuda_compilable(x, y, BLOCK_SIZE)
        else:
            kernel = KernelRegistry.get_kernel(_KERNEL_NAME)
            output = kernel(x, y, BLOCK_SIZE)

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        return output_grad, None


def add_scalar_cuda(x: torch.Tensor, y: float) -> torch.Tensor:
    """tensor addition

    Args:
        x (torch.Tensor): input tensor
        y (float): input scalar

    Returns:
        torch.Tensor: output tensor
    """

    return _AddScalar_CUDA.apply(x, y)
