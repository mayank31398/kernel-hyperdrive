import torch

from ...constants import LIBRARY_NAME
from ...kernel_registry import KernelRegistry


_KERNEL_NAME = "vector_addition_forward_cuda"
BLOCK_SIZE = 1024


# non compilable kernel
def _vector_addition_forward_cuda(x: torch.Tensor, y: torch.Tensor, memory_efficient: bool) -> torch.Tensor:
    if not hasattr(_vector_addition_forward_cuda, "_kernel"):
        _vector_addition_forward_cuda._kernel = KernelRegistry.get_kernel(_KERNEL_NAME)

    return _vector_addition_forward_cuda._kernel(x, y, memory_efficient, BLOCK_SIZE)


# compilable non memory efficient kernel
@torch.library.custom_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args=())
def _vector_addition_forward_cuda_compilable(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _vector_addition_forward_cuda(x, y, memory_efficient=False, BLOCK_SIZE=BLOCK_SIZE)


# compilable memory efficient kernel
@torch.library.custom_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}-memory-efficient", mutates_args=("x"))
def _vector_addition_forward_cuda_compilable_memory_efficient(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _vector_addition_forward_cuda(x, y, memory_efficient=True, BLOCK_SIZE=BLOCK_SIZE)


def _fake(x: torch.Tensor, y: torch.Tensor, memory_efficient: bool) -> torch.Tensor:
    return torch.empty_like(x)


_vector_addition_forward_cuda_compilable.register_fake(_fake)
_vector_addition_forward_cuda_compilable_memory_efficient.register_fake(_fake)


class _VectorAddition_CUDA(torch.autograd.Function):
    @torch.profiler.record_function(f"{LIBRARY_NAME}:{_KERNEL_NAME}")
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor, memory_efficient: bool) -> torch.Tensor:
        if torch.compiler.is_compiling():
            if memory_efficient:
                output = _vector_addition_forward_cuda_compilable_memory_efficient(x, y)
            else:
                output = _vector_addition_forward_cuda_compilable(x, y)
        else:
            output = _vector_addition_forward_cuda(x, y, memory_efficient=memory_efficient)

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        return output_grad, output_grad, None


def vector_addition_cuda(x: torch.Tensor, y: torch.Tensor, memory_efficient: bool = False) -> torch.Tensor:
    """vector addition

    Args:
        x (torch.Tensor): input tensor
        y (torch.Tensor): input tensor
        memory_efficient (bool, optional): whether to do an in-place op, will modify `x` if set to True. Defaults to False.

    Returns:
        torch.Tensor: output tensor
    """

    return _VectorAddition_CUDA.apply(x, y, memory_efficient)
