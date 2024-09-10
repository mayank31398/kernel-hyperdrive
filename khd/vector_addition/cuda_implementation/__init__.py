import torch

from ...kernel_registry import KernelRegistry


_KERNEL_NAME = "vector_addition_forward_cuda"


class _VectorAddition_CUDA(torch.autograd.Function):
    @torch.profiler.record_function(_KERNEL_NAME)
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor, memory_efficient: bool) -> torch.Tensor:
        if not hasattr(_VectorAddition_CUDA.forward, "_kernel"):
            _VectorAddition_CUDA.forward._kernel = KernelRegistry.get_kernel(_KERNEL_NAME)

        return _VectorAddition_CUDA.forward._kernel(x, y, memory_efficient, 1024)

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
