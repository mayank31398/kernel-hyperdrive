import torch

from ...kernel_registry import KernelRegistry


_KERNEL_NAME = "vector_addition_forward"


def _cuda_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if not hasattr(_cuda_kernel, "_kernel"):
        _cuda_kernel._kernel = KernelRegistry.get_kernel(_KERNEL_NAME)

    return _cuda_kernel._kernel(x, y)


# this registers the kernel with PyTorch to make it work with torch.compile
@torch.library.custom_op(f"khd::{_KERNEL_NAME}", mutates_args=())
def vector_addition_forward_cuda_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _cuda_kernel(x, y)


# this tells torch.compile the output shape given the input shape
@vector_addition_forward_cuda_kernel.register_fake
def _(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


class _VectorAddition_CUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return vector_addition_forward_cuda_kernel(x, y)

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return output_grad, output_grad


def vector_addition_cuda(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _VectorAddition_CUDA.apply(x, y)
