import torch
import triton

from .kernels import swiglu_backward_triton_kernel, swiglu_forward_triton_kernel


class _Swiglu_Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        assert gate.is_cuda, "tensor gate is not on GPU"
        assert up.is_cuda, "tensor up is not on GPU"

        assert gate.size() == up.size(), "tensors gate and up should have same shape"
        assert gate.type() == up.type(), "tensors gate and up should have same dtype"

        ctx.save_for_backward(gate, up)

        output = torch.empty_like(gate)

        num_elements = gate.numel()
        grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

        swiglu_forward_triton_kernel[grid](gate.view(-1), up.view(-1), output, num_elements, BLOCK_SIZE=1024)

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate, up = ctx.saved_tensors

        num_elements = output_grad.numel()
        grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

        # the kernel uses the gate and up tensors to store the gradients in-place for memory savings
        swiglu_backward_triton_kernel[grid](
            gate.view(-1), up.view(-1), output_grad.view(-1), num_elements, BLOCK_SIZE=1024
        )

        return gate, up


def swiglu_triton(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return _Swiglu_Triton.apply(gate, up)
