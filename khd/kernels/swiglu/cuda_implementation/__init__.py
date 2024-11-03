import torch

from ....constants import LIBRARY_NAME
from ....kernel_registry import KernelRegistry


_FORWARD_KERNEL_NAME = "swiglu_forward_cuda"
_BACKWARD_KERNEL_NAME = "swiglu_backward_cuda"


@torch.library.custom_op(f"{LIBRARY_NAME}::{_FORWARD_KERNEL_NAME}", mutates_args={"output"})
def _swiglu_forward_cuda(gate: torch.Tensor, up: torch.Tensor, output: torch.Tensor, BLOCK_SIZE: int) -> None:
    KernelRegistry.get_kernel(_FORWARD_KERNEL_NAME)(gate, up, output, BLOCK_SIZE)


@torch.library.custom_op(f"{LIBRARY_NAME}::{_BACKWARD_KERNEL_NAME}", mutates_args={"gate_grad", "up_grad"})
def _swiglu_backward_cuda(
    gate: torch.Tensor,
    up: torch.Tensor,
    output_grad: torch.Tensor,
    gate_grad: torch.Tensor,
    up_grad: torch.Tensor,
    BLOCK_SIZE: int,
) -> None:
    KernelRegistry.get_kernel(_FORWARD_KERNEL_NAME)(gate, up, output_grad, gate_grad, up_grad, BLOCK_SIZE)


class _Swiglu_CUDA(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, gate: torch.Tensor, up: torch.Tensor, BLOCK_SIZE_forward: int, BLOCK_SIZE_backward: int
    ) -> torch.Tensor:
        assert gate.is_cuda, "tensor gate is not on GPU"
        assert up.is_cuda, "tensor up is not on GPU"

        assert gate.size() == up.size(), "tensors gate and up should have same shape"
        assert gate.type() == up.type(), "tensors gate and up should have same dtype"

        ctx.save_for_backward(gate, up)
        ctx.BLOCK_SIZE_backward = BLOCK_SIZE_backward

        output = torch.empty_like(gate)

        _swiglu_forward_cuda(gate, up, output, BLOCK_SIZE_forward)

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        gate, up = ctx.saved_tensors

        gate_grad = torch.empty_like(gate)
        up_grad = torch.empty_like(up)

        _swiglu_backward_cuda(gate, up, output_grad, gate_grad, up_grad, ctx.BLOCK_SIZE_backward)

        return gate_grad, up_grad, None, None


def swiglu_cuda(
    gate: torch.Tensor, up: torch.Tensor, BLOCK_SIZE_forward: int, BLOCK_SIZE_backward: int
) -> torch.Tensor:
    """swiglu

    Args:
        gate (torch.Tensor): gate tensor
        up (torch.Tensor): up tensor
        BLOCK_SIZE_forward (int): forward block size
        BLOCK_SIZE_backward (int): backward block size

    Returns:
        torch.Tensor: output tensor
    """

    return _Swiglu_CUDA.apply(gate, up, BLOCK_SIZE_forward, BLOCK_SIZE_backward)
