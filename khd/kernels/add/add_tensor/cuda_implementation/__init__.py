import torch

from .....constants import LIBRARY_NAME
from .....kernel_registry import KernelRegistry
from .....utils import CutoTune, get_default_cuda_autotune_configs
from ....utils import torch_custom_op


_KERNEL_NAME = "add_tensor_forward_cuda"


def _add_tensor_forward_cuda(
    x: torch.Tensor,
    y: torch.Tensor,
    output: torch.Tensor,
    vectorized_loop_size: int,
    BLOCK_SIZE: int,
) -> None:
    KernelRegistry.get_kernel(_KERNEL_NAME)(x, y, output, vectorized_loop_size, BLOCK_SIZE)


@torch_custom_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"output"})
def _add_tensor_forward_cuda_compilable(
    x: torch.Tensor,
    y: torch.Tensor,
    output: torch.Tensor,
    vectorized_loop_size: int,
    BLOCK_SIZE: int,
) -> None:
    _add_tensor_forward_cuda(x=x, y=y, output=output, vectorized_loop_size=vectorized_loop_size, BLOCK_SIZE=BLOCK_SIZE)


@CutoTune(
    configs=get_default_cuda_autotune_configs(extra_config_condition=lambda kwargs: kwargs["x"].dtype == torch.float32)
)
def add_tensor_forward_cuda(
    x: torch.Tensor,
    y: torch.Tensor,
    output: torch.Tensor,
    vectorized_loop_size: int,
    BLOCK_SIZE: int,
) -> None:
    function = _add_tensor_forward_cuda_compilable if torch.compiler.is_compiling() else _add_tensor_forward_cuda
    function(x=x, y=y, output=output, vectorized_loop_size=vectorized_loop_size, BLOCK_SIZE=BLOCK_SIZE)
