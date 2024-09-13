import torch

from ...constants import LIBRARY_NAME
from ...kernel_registry import KernelRegistry


_KERNEL_NAME = "vector_addition_forward_cuda"
BLOCK_SIZE = 1024


@torch.library.custom_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={})
def _vector_addition_forward_cuda_compilable(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return KernelRegistry.get_kernel(_KERNEL_NAME)(x, y, False, BLOCK_SIZE)


# compilable memory efficient kernel
@torch.library.custom_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}_memory_efficient", mutates_args={"x"})
def _vector_addition_forward_cuda_compilable_memory_efficient(x: torch.Tensor, y: torch.Tensor) -> None:
    KernelRegistry.get_kernel(_KERNEL_NAME)(x, y, True, BLOCK_SIZE)


@_vector_addition_forward_cuda_compilable.register_fake
def _fake_vector_addition_forward_cuda_compilable(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@_vector_addition_forward_cuda_compilable_memory_efficient.register_fake
def _fake_vector_addition_forward_cuda_compilable_memory_efficient(x: torch.Tensor, y: torch.Tensor) -> None:
    return


class _VectorAddition_CUDA(torch.autograd.Function):
    @torch.profiler.record_function(f"{LIBRARY_NAME}:{_KERNEL_NAME}")
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor, memory_efficient: bool) -> torch.Tensor:
        if torch.compiler.is_compiling():
            if memory_efficient:
                _vector_addition_forward_cuda_compilable_memory_efficient(x, y)
                output = x
            else:
                output = _vector_addition_forward_cuda_compilable(x, y)
        else:
            if memory_efficient:
                KernelRegistry.get_kernel(_KERNEL_NAME)(x, y, memory_efficient, BLOCK_SIZE)
                output = x
            else:
                output = KernelRegistry.get_kernel(_KERNEL_NAME)(x, y, memory_efficient, BLOCK_SIZE)

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
