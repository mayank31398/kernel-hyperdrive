import torch

from ....constants import LIBRARY_NAME
from ....kernel_registry import KernelRegistry
from ...utils import torch_custom_op


KernelRegistry.get_kernel("add_tensor_forward_cuda")


_BACKWARD_KERNEL_NAME = "swiglu_backward_cuda"


@torch_custom_op(f"{LIBRARY_NAME}::{_BACKWARD_KERNEL_NAME}", mutates_args={})
def swiglu_backward_cuda(
    gate: torch.Tensor, up: torch.Tensor, output_grad: torch.Tensor, BLOCK_SIZE: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return KernelRegistry.get_kernel(_BACKWARD_KERNEL_NAME)(gate, up, output_grad, BLOCK_SIZE)


@swiglu_backward_cuda.register_fake
def _(
    gate: torch.Tensor, up: torch.Tensor, output_grad: torch.Tensor, BLOCK_SIZE: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(gate), torch.empty_like(gate)


class _Swiglu_CUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor, BLOCK_SIZE_forward: int) -> torch.Tensor:
        assert gate.is_cuda, "tensor gate is not on GPU"
        assert up.is_cuda, "tensor up is not on GPU"

        assert gate.size() == up.size(), "tensors gate and up should have same shape"
        assert gate.type() == up.type(), "tensors gate and up should have same dtype"

        ctx.save_for_backward(gate, up)

        output = torch.empty_like(gate)

        output = torch.ops.khd.swiglu_forward_cuda(gate, up, BLOCK_SIZE_forward)

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        gate, up = ctx.saved_tensors

        BLOCK_SIZE = 1024

        gate_grad, up_grad = torch.ops.khd.swiglu_backward_cuda(gate, up, output_grad, BLOCK_SIZE)

        return gate_grad, up_grad


def swiglu_cuda(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """swiglu

    Args:
        gate (torch.Tensor): gate tensor
        up (torch.Tensor): up tensor

    Returns:
        torch.Tensor: output tensor
    """

    return _Swiglu_CUDA.apply(gate, up)
