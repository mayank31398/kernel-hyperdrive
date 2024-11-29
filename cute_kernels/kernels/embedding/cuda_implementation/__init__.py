import torch

from ....constants import LIBRARY_NAME
from ....kernel_registry import KernelRegistry
from ....utils import torch_custom_op


_FORWARD_KERNEL_NAME = "embedding_forward_cuda"
_BACKWARD_KERNEL_NAME = "embedding_backward_cuda"


@torch_custom_op(f"{LIBRARY_NAME}::{_FORWARD_KERNEL_NAME}", mutates_args={"output"})
def embedding_forward_cuda_kernel(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    vector_instruction_width: int,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> None:
    KernelRegistry.get_kernel(_FORWARD_KERNEL_NAME)(
        input_ids, weight, output, vector_instruction_width, BLOCK_SIZE_B, BLOCK_SIZE_H
    )


@torch_custom_op(f"{LIBRARY_NAME}::{_BACKWARD_KERNEL_NAME}", mutates_args={"gate_grad", "up_grad"})
def embedding_backward_cuda_kernel(
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
