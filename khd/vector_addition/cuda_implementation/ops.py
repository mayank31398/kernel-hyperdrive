import torch

from ...constants import LIBRARY_NAME
from ...kernel_registry import KernelRegistry


_KERNEL_NAME = "vector_addition_forward_cuda"


@torch.library.custom_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"output"})
def _vector_addition_forward_cuda_compilable(
    x: torch.Tensor, y: torch.Tensor, output: torch.Tensor, num_elements: int, BLOCK_SIZE: int
) -> None:
    KernelRegistry.get_kernel(_KERNEL_NAME)(x, y, output, num_elements, BLOCK_SIZE)


@_vector_addition_forward_cuda_compilable.register_fake
def _(x: torch.Tensor, y: torch.Tensor, output: torch.Tensor, num_elements: int, BLOCK_SIZE: int) -> None:
    return


class _VectorAddition_CUDA(torch.autograd.Function):
    @torch.profiler.record_function(f"{LIBRARY_NAME}:{_KERNEL_NAME}")
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "tensor x is not on GPU"
        assert y.is_cuda, "tensor y is not on GPU"

        assert x.size() == y.size(), "tensors x and y should have same shape"
        assert x.type() == y.type(), "tensors x and y should have same dtype"

        output = torch.empty_like(x)

        num_elements = x.numel()

        BLOCK_SIZE = 1024

        if torch.compiler.is_compiling():
            _vector_addition_forward_cuda_compilable(x, y, output, num_elements, BLOCK_SIZE)
        else:
            kernel = KernelRegistry.get_kernel(_KERNEL_NAME)
            kernel(x, y, output, num_elements, BLOCK_SIZE)

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        return output_grad, output_grad


def vector_addition_cuda(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """vector addition

    Args:
        x (torch.Tensor): input tensor
        y (torch.Tensor): input tensor

    Returns:
        torch.Tensor: output tensor
    """

    return _VectorAddition_CUDA.apply(x, y)
