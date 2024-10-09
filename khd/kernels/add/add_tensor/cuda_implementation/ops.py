import torch

from .....constants import LIBRARY_NAME
from .....kernel_registry import KernelRegistry
from ....utils import torch_custom_op


_KERNEL_NAME = "add_tensor_forward_cuda"


@torch_custom_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={})
def _add_tensor_forward_cuda_compilable(
    x: torch.Tensor, y: torch.Tensor, use_efficient_kernel: bool, BLOCK_SIZE: int
) -> torch.Tensor:
    return KernelRegistry.get_kernel(_KERNEL_NAME)(x, y, use_efficient_kernel, BLOCK_SIZE)


@_add_tensor_forward_cuda_compilable.register_fake
def _(x: torch.Tensor, y: torch.Tensor, use_efficient_kernel: bool, BLOCK_SIZE: int) -> torch.Tensor:
    return torch.empty_like(x)


class _AddTensor_CUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor, use_efficient_kernel: bool) -> torch.Tensor:
        BLOCK_SIZE = 1024

        if torch.compiler.is_compiling():
            output = _add_tensor_forward_cuda_compilable(x, y, use_efficient_kernel, BLOCK_SIZE)
        else:
            kernel = KernelRegistry.get_kernel(_KERNEL_NAME)
            output = kernel(x, y, use_efficient_kernel, BLOCK_SIZE)

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        return output_grad, output_grad


def add_tensor_cuda(x: torch.Tensor, y: torch.Tensor, use_efficient_kernel: bool = True) -> torch.Tensor:
    """tensor addition

    Args:
        x (torch.Tensor): input tensor
        y (torch.Tensor): input tensor
        use_efficient_kernel (bool): whether to use the efficient CUDA kernel

    Returns:
        torch.Tensor: output tensor
    """

    return _AddTensor_CUDA.apply(x, y, use_efficient_kernel)
