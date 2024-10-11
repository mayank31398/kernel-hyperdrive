import torch

from .....constants import LIBRARY_NAME
from .....kernel_registry import KernelRegistry
from .....utils import AutoTune, get_vectorized_autotune_configs
from ....utils import torch_custom_op


_KERNEL_NAME = "add_tensor_forward_cuda"


@AutoTune(configs=get_vectorized_autotune_configs(torch.float32))
def _add_tensor_forward_cuda_autotuned_fp32(
    x: torch.Tensor, y: torch.Tensor, vectorized_loop_size: int, BLOCK_SIZE: int
) -> torch.Tensor:
    return KernelRegistry.get_kernel(_KERNEL_NAME)(x, y, vectorized_loop_size, BLOCK_SIZE)


@AutoTune(configs=get_vectorized_autotune_configs(torch.float16))
def _add_tensor_forward_cuda_autotuned_fp16(
    x: torch.Tensor, y: torch.Tensor, vectorized_loop_size: int, BLOCK_SIZE: int
) -> torch.Tensor:
    return KernelRegistry.get_kernel(_KERNEL_NAME)(x, y, vectorized_loop_size, BLOCK_SIZE)


def _add_tensor_forward_cuda(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.dtype in [torch.float16, torch.bfloat16]:
        output = _add_tensor_forward_cuda_autotuned_fp16(x, y)
    elif x.dtype == torch.float32:
        output = _add_tensor_forward_cuda_autotuned_fp32(x, y)

    return output


@torch_custom_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={})
def _add_tensor_forward_cuda_compilable(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _add_tensor_forward_cuda(x, y)


@_add_tensor_forward_cuda_compilable.register_fake
def _(x: torch.Tensor, y: torch.Tensor, vectorized_loop_size: int, BLOCK_SIZE: int) -> torch.Tensor:
    return torch.empty_like(x)
