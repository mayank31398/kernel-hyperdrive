import os

import torch

from ...kernel_registry import get_kernel, register_kernel


_KERNEL_NAME = "vector_addition_forward"

register_kernel(
    _KERNEL_NAME,
    sources=[
        os.path.join(os.path.dirname(__file__), "vector_addition.cpp"),
        os.path.join(os.path.dirname(__file__), "vector_addition.cu"),
    ],
)


class _VectorAddition_CUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        global _KERNEL_NAME

        function = get_kernel(_KERNEL_NAME)
        return function(x, y)

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return output_grad, output_grad


# this registers the kernel with PyTorch to make it work with torch.compile
@torch.library.custom_op("khd::vector_addition_cuda", mutates_args=())
def vector_addition_cuda(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _VectorAddition_CUDA.apply(x, y)


# this tells torch.compile the output shape given the input shape
@vector_addition_cuda.register_fake
def _(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)
