import torch

from ....constants import LIBRARY_NAME
from ....kernel_registry import KernelRegistry
from ....utils import cute_op


_FORWARD_KERNEL_NAME = "swiglu_forward_cuda"
_BACKWARD_KERNEL_NAME = "swiglu_backward_cuda"


@cute_op(f"{LIBRARY_NAME}::{_FORWARD_KERNEL_NAME}", mutates_args={"output"})
def swiglu_forward_cuda(
    gate: torch.Tensor, up: torch.Tensor, output: torch.Tensor, vector_instruction_width: int, BLOCK_SIZE: int
) -> None:
    KernelRegistry.get_kernel(_FORWARD_KERNEL_NAME)(gate, up, output, vector_instruction_width, BLOCK_SIZE)


@cute_op(f"{LIBRARY_NAME}::{_BACKWARD_KERNEL_NAME}", mutates_args={"gate_grad", "up_grad"})
def swiglu_backward_cuda(
    gate: torch.Tensor,
    up: torch.Tensor,
    output_grad: torch.Tensor,
    gate_grad: torch.Tensor,
    up_grad: torch.Tensor,
    vector_instruction_width: int,
    BLOCK_SIZE: int,
) -> None:
    KernelRegistry.get_kernel(_BACKWARD_KERNEL_NAME)(
        gate, up, output_grad, gate_grad, up_grad, vector_instruction_width, BLOCK_SIZE
    )
