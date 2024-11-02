import torch

from ....constants import LIBRARY_NAME
from ....kernel_registry import KernelRegistry
from ...utils import torch_custom_op


_FORWARD_KERNEL_NAME = "swiglu_forward_cuda"
_BACKWARD_KERNEL_NAME = "swiglu_backward_cuda"


def _swiglu_forward_cuda(gate: torch.Tensor, up: torch.Tensor, BLOCK_SIZE: int) -> torch.Tensor:
    return KernelRegistry.get_kernel(_FORWARD_KERNEL_NAME)(gate, up, BLOCK_SIZE)


@torch_custom_op(f"{LIBRARY_NAME}::{_FORWARD_KERNEL_NAME}", mutates_args={})
def _swiglu_forward_cuda_compilable(gate: torch.Tensor, up: torch.Tensor, BLOCK_SIZE: int) -> torch.Tensor:
    return _swiglu_forward_cuda(gate, up, BLOCK_SIZE)


@_swiglu_forward_cuda_compilable.register_fake
def _(gate: torch.Tensor, up: torch.Tensor, BLOCK_SIZE: int) -> torch.Tensor:
    return torch.empty_like(gate)


@torch_custom_op(f"{LIBRARY_NAME}::{_BACKWARD_KERNEL_NAME}", mutates_args={})
def _swiglu_backward_cuda(
    gate: torch.Tensor, up: torch.Tensor, output_grad: torch.Tensor, BLOCK_SIZE: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return KernelRegistry.get_kernel(_BACKWARD_KERNEL_NAME)(gate, up, output_grad, BLOCK_SIZE)


@torch_custom_op(f"{LIBRARY_NAME}::{_BACKWARD_KERNEL_NAME}", mutates_args={})
def _swiglu_backward_cuda_compilable(
    gate: torch.Tensor, up: torch.Tensor, output_grad: torch.Tensor, BLOCK_SIZE: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return _swiglu_backward_cuda(gate, up, output_grad, BLOCK_SIZE)


@_swiglu_backward_cuda_compilable.register_fake
def _(
    gate: torch.Tensor, up: torch.Tensor, output_grad: torch.Tensor, BLOCK_SIZE: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(gate), torch.empty_like(gate)
