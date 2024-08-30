import torch

from ...kernel_registry import KernelRegistry


_KERNEL_NAME = "vector_addition_forward"


class _VectorAddition_CUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not hasattr(_VectorAddition_CUDA.forward, "_kernel"):
            _VectorAddition_CUDA.forward._kernel = KernelRegistry.get_kernel(_KERNEL_NAME)

        return _VectorAddition_CUDA.forward._kernel(x, y)

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return output_grad, output_grad


def vector_addition_cuda(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _VectorAddition_CUDA.apply(x, y)
