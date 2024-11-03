import torch

from ....kernel_registry import KernelRegistry


KernelRegistry.get_kernel("swiglu_forward_cuda")
KernelRegistry.get_kernel("swiglu_backward_cuda")


class _Swiglu_CUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor, BLOCK_SIZE_forward: int) -> torch.Tensor:
        assert gate.is_cuda, "tensor gate is not on GPU"
        assert up.is_cuda, "tensor up is not on GPU"

        assert gate.size() == up.size(), "tensors gate and up should have same shape"
        assert gate.type() == up.type(), "tensors gate and up should have same dtype"

        ctx.save_for_backward(gate, up)

        output = torch.empty_like(gate)

        torch.ops.khd.swiglu_forward_cuda(gate, up, output, BLOCK_SIZE_forward)

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        gate, up = ctx.saved_tensors

        BLOCK_SIZE = 1024

        gate_grad = torch.empty_like(gate)
        up_grad = torch.empty_like(up)

        gate_grad, up_grad = torch.ops.khd.swiglu_backward_cuda(gate, up, output_grad, gate_grad, up_grad, BLOCK_SIZE)

        return gate_grad, up_grad, None


def swiglu_cuda(gate: torch.Tensor, up: torch.Tensor, BLOCK_SIZE_forward: int) -> torch.Tensor:
    """swiglu

    Args:
        gate (torch.Tensor): gate tensor
        up (torch.Tensor): up tensor
        BLOCK_SIZE_forward (int): forward block size

    Returns:
        torch.Tensor: output tensor
    """

    return _Swiglu_CUDA.apply(gate, up, BLOCK_SIZE_forward)
