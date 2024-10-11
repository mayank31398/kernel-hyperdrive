import torch

from .....constants import LIBRARY_NAME
from .....kernel_registry import KernelRegistry
from .....utils import AutoTune, get_vectorized_autotune_configs
from ....utils import torch_custom_op


_KERNEL_NAME = "add_tensor_forward_cuda"


@AutoTune(configs=get_vectorized_autotune_configs(), trigger_keys=["x", "y", "dtype"])
def _add_tensor_forward_cuda_autotuned(
    x: torch.Tensor, y: torch.Tensor, dtype: torch.dtype, vectorized_loop_size: int, BLOCK_SIZE: int
) -> torch.Tensor:
    return KernelRegistry.get_kernel(_KERNEL_NAME)(x, y, vectorized_loop_size, BLOCK_SIZE)


def _add_tensor_forward_cuda(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _add_tensor_forward_cuda_autotuned(x, y, x.dtype)


@torch_custom_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={})
def _add_tensor_forward_cuda_compilable(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _add_tensor_forward_cuda(x, y)


@_add_tensor_forward_cuda_compilable.register_fake
def _(x: torch.Tensor, y: torch.Tensor, vectorized_loop_size: int, BLOCK_SIZE: int) -> torch.Tensor:
    return torch.empty_like(x)


class _AddTensor_CUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if torch.compiler.is_compiling():
            output = _add_tensor_forward_cuda_compilable(x, y)
        else:
            output = _add_tensor_forward_cuda(x, y)

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        return output_grad, output_grad, None


def add_tensor_cuda(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """tensor addition

    Args:
        x (torch.Tensor): input tensor
        y (torch.Tensor): input tensor

    Returns:
        torch.Tensor: output tensor
    """

    return _AddTensor_CUDA.apply(x, y)
