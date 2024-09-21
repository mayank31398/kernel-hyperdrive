import torch
import triton

from ...constants import LIBRARY_NAME
from .kernels import swiglu_backward_triton_kernel, swiglu_forward_triton_kernel


_FORWARD_KERNEL_NAME = "swiglu_forward_triton_kernel"
_BACKWARD_KERNEL_NAME = "swiglu_backward_triton_kernel"


class _Swiglu_Triton(torch.autograd.Function):
    @torch.profiler.record_function(f"{LIBRARY_NAME}:{_FORWARD_KERNEL_NAME}")
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

        BLOCK_SIZE = 1024

        swiglu_forward_triton_kernel[grid](
            gate_ptr=gate.view(-1),
            up_ptr=up.view(-1),
            output_ptr=output.view(-1),
            num_elements=num_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return output

    @torch.profiler.record_function(f"{LIBRARY_NAME}:{_BACKWARD_KERNEL_NAME}")
    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        gate, up = ctx.saved_tensors

        num_elements = output_grad.numel()
        grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

        gate_grad = torch.empty_like(gate)
        up_grad = torch.empty_like(up)

        BLOCK_SIZE = 1024

        # the kernel uses the gate and up tensors to store the gradients in-place for memory savings
        swiglu_backward_triton_kernel[grid](
            gate_ptr=gate.view(-1),
            up_ptr=up.view(-1),
            output_grad_ptr=output_grad.view(-1),
            gate_grad_ptr=gate_grad.view(-1),
            up_grad_ptr=up_grad.view(-1),
            num_elements=num_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return gate_grad, up_grad, None


def swiglu_triton(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """swiglu

    Args:
        x (torch.Tensor): input tensor
        y (torch.Tensor): input tensor

    Returns:
        torch.Tensor: output tensor
    """

    return _Swiglu_Triton.apply(gate, up)
