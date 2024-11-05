import torch

from ....constants import LIBRARY_NAME
from ....kernel_registry import KernelRegistry


_FORWARD_KERNEL_NAME = "swiglu_forward_cuda"
_BACKWARD_KERNEL_NAME = "swiglu_backward_cuda"


def swiglu_forward_cuda_kernel(gate: torch.Tensor, up: torch.Tensor, output: torch.Tensor, BLOCK_SIZE: int) -> None:
    KernelRegistry.get_kernel(_FORWARD_KERNEL_NAME)(gate, up, output, BLOCK_SIZE)


@torch.library.custom_op(f"{LIBRARY_NAME}::{_FORWARD_KERNEL_NAME}", mutates_args={"output"})
def swiglu_forward_cuda_kernel_compileable(
    gate: torch.Tensor, up: torch.Tensor, output: torch.Tensor, BLOCK_SIZE: int
) -> None:
    swiglu_forward_cuda_kernel(gate=gate, up=up, output=output, BLOCK_SIZE=BLOCK_SIZE)


def swiglu_backward_cuda_kernel(
    gate: torch.Tensor,
    up: torch.Tensor,
    output_grad: torch.Tensor,
    gate_grad: torch.Tensor,
    up_grad: torch.Tensor,
    BLOCK_SIZE: int,
) -> None:
    KernelRegistry.get_kernel(_BACKWARD_KERNEL_NAME)(gate, up, output_grad, gate_grad, up_grad, BLOCK_SIZE)


@torch.library.custom_op(f"{LIBRARY_NAME}::{_BACKWARD_KERNEL_NAME}", mutates_args={"gate_grad", "up_grad"})
def swiglu_backward_cuda_kernel_compileable(
    gate: torch.Tensor,
    up: torch.Tensor,
    output_grad: torch.Tensor,
    gate_grad: torch.Tensor,
    up_grad: torch.Tensor,
    BLOCK_SIZE: int,
) -> None:
    swiglu_backward_cuda_kernel(
        gate=gate, up=up, output_grad=output_grad, gate_grad=gate_grad, up_grad=up_grad, BLOCK_SIZE=BLOCK_SIZE
    )
