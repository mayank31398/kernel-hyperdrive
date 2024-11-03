import torch
import triton

from .kernels import swiglu_backward_triton_kernel, swiglu_forward_triton_kernel


class _Swiglu_Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor, BLOCK_SIZE_forward: int) -> torch.Tensor:
        assert gate.is_cuda, "tensor gate is not on GPU"
        assert up.is_cuda, "tensor up is not on GPU"

        assert gate.size() == up.size(), "tensors gate and up should have same shape"
        assert gate.type() == up.type(), "tensors gate and up should have same dtype"

        ctx.save_for_backward(gate, up)

        output = torch.empty_like(gate)

        num_elements = gate.numel()
        grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

        with torch.device(gate.device):
            swiglu_forward_triton_kernel[grid](
                gate_ptr=gate,
                up_ptr=up,
                output_ptr=output,
                num_elements=num_elements,
                BLOCK_SIZE=BLOCK_SIZE_forward,
            )

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        gate, up = ctx.saved_tensors

        num_elements = output_grad.numel()
        grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

        gate_grad = torch.empty_like(gate)
        up_grad = torch.empty_like(up)

        BLOCK_SIZE = 1024

        with torch.device(gate.device):
            swiglu_backward_triton_kernel[grid](
                gate_ptr=gate,
                up_ptr=up,
                output_grad_ptr=output_grad,
                gate_grad_ptr=gate_grad,
                up_grad_ptr=up_grad,
                num_elements=num_elements,
                BLOCK_SIZE=BLOCK_SIZE,
            )

        return gate_grad, up_grad


def swiglu_triton(gate: torch.Tensor, up: torch.Tensor, BLOCK_SIZE_forward: int) -> torch.Tensor:
    """swiglu

    Args:
        x (torch.Tensor): input tensor
        y (torch.Tensor): input tensor
        BLOCK_SIZE_forward (int): forward block size

    Returns:
        torch.Tensor: output tensor
    """

    return _Swiglu_Triton.apply(gate, up, BLOCK_SIZE_forward)
