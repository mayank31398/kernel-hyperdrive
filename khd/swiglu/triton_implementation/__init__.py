import torch
import triton

from .kernels import swiglu_backward_triton_kernel, swiglu_forward_triton_kernel


class _Swiglu_Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor, in_place: bool) -> torch.Tensor:
        assert gate.is_cuda, "tensor gate is not on GPU"
        assert up.is_cuda, "tensor up is not on GPU"

        assert gate.size() == up.size(), "tensors gate and up should have same shape"
        assert gate.type() == up.type(), "tensors gate and up should have same dtype"

        ctx.save_for_backward(gate, up)
        ctx.in_place = in_place

        output = torch.empty_like(gate)

        num_elements = gate.numel()
        grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

        swiglu_forward_triton_kernel[grid](
            gate_ptr=gate.view(-1), up_ptr=up.view(-1), output_ptr=output, num_elements=num_elements, BLOCK_SIZE=1024
        )

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        gate, up = ctx.saved_tensors
        in_place = ctx.in_place

        num_elements = output_grad.numel()
        grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

        gate_grad = gate if in_place else torch.empty_like(gate)
        up_grad = up if in_place else torch.empty_like(up)

        # the kernel uses the gate and up tensors to store the gradients in-place for memory savings
        swiglu_backward_triton_kernel[grid](
            gate_ptr=gate.view(-1),
            up_ptr=up.view(-1),
            output_grad_ptr=output_grad.view(-1),
            gate_grad_ptr=gate_grad.view(-1),
            up_grad_ptr=up_grad.view(-1),
            num_elements=num_elements,
            BLOCK_SIZE=1024,
        )

        return gate_grad, up_grad, None


def swiglu_triton(gate: torch.Tensor, up: torch.Tensor, in_place: bool = False) -> torch.Tensor:
    """swiglu

    Args:
        x (torch.Tensor): input tensor
        y (torch.Tensor): input tensor
        in_place (bool, optional): whether to do an in-place op, uses `gate` and `up` to store gradients in the
            backward pass if set to True. Defaults to False.

    Returns:
        torch.Tensor: output tensor
    """

    return _Swiglu_Triton.apply(gate, up, in_place)
