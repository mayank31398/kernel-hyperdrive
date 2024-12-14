import torch

from ....constants import LIBRARY_NAME
from ....utils import ceil_divide, cute_op
from .kernels_backward import swiglu_unchunked_backward_triton_kernel
from .kernels_forward import swiglu_unchunked_forward_triton_kernel


_FORWARD_KERNEL_NAME = "swiglu_unchunked_forward_triton"
_BACKWARD_KERNEL_NAME = "swiglu_unchunked_backward_triton"


@cute_op(f"{LIBRARY_NAME}::{_FORWARD_KERNEL_NAME}", mutates_args={"output"})
def swiglu_forward_triton(x: torch.Tensor, output: torch.Tensor, BLOCK_SIZE_B: int, BLOCK_SIZE_H: int) -> None:
    H = x.size(-1)
    B = x.numel() // H

    with torch.device(x.device):
        swiglu_unchunked_forward_triton_kernel[(ceil_divide(B, BLOCK_SIZE_B), ceil_divide(H, BLOCK_SIZE_H))](
            x_ptr=x,
            output_ptr=output,
            B=B,
            H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )


@cute_op(f"{LIBRARY_NAME}::{_BACKWARD_KERNEL_NAME}", mutates_args={"gate_grad", "up_grad"})
def swiglu_backward_triton(
    gate: torch.Tensor,
    up: torch.Tensor,
    output_grad: torch.Tensor,
    gate_grad: torch.Tensor,
    up_grad: torch.Tensor,
    BLOCK_SIZE: int,
) -> None:
    num_elements = gate.numel()

    with torch.device(gate.device):
        swiglu_backward_triton_kernel[(ceil_divide(num_elements, BLOCK_SIZE),)](
            gate_ptr=gate,
            up_ptr=up,
            output_grad_ptr=output_grad,
            gate_grad_ptr=gate_grad,
            up_grad_ptr=up_grad,
            num_elements=num_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
