import torch
import triton

from ...utils import LibaryRecordFunction
from .kernels import swiglu_backward_triton_kernel, swiglu_forward_triton_kernel


_FORWARD_KERNEL_NAME = "swiglu_forward_triton_kernel"
_BACKWARD_KERNEL_NAME = "swiglu_backward_triton_kernel"
FORWARD_BLOCK_SIZE = 1024
BACKWARD_BLOCK_SIZE = 1024


class _Swiglu_Triton(torch.autograd.Function):
    @LibaryRecordFunction(_FORWARD_KERNEL_NAME)
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor, memory_efficient: bool) -> torch.Tensor:
        assert gate.is_cuda, "tensor gate is not on GPU"
        assert up.is_cuda, "tensor up is not on GPU"

        assert gate.size() == up.size(), "tensors gate and up should have same shape"
        assert gate.type() == up.type(), "tensors gate and up should have same dtype"

        if memory_efficient:
            if gate.is_leaf or up.is_leaf:
                # forward pass if fine but backward pass will be incorrect due to in-place ops
                # we raise error in forward pass though
                raise RuntimeError("leaf variables can't be used in an in-place operation")

        ctx.save_for_backward(gate, up)
        ctx.memory_efficient = memory_efficient

        output = torch.empty_like(gate)

        num_elements = gate.numel()
        grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

        swiglu_forward_triton_kernel[grid](
            gate_ptr=gate.view(-1),
            up_ptr=up.view(-1),
            output_ptr=output.view(-1),
            num_elements=num_elements,
            BLOCK_SIZE=FORWARD_BLOCK_SIZE,
        )

        return output

    @LibaryRecordFunction(_BACKWARD_KERNEL_NAME)
    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        gate, up = ctx.saved_tensors
        memory_efficient = ctx.memory_efficient

        num_elements = output_grad.numel()
        grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

        gate_grad = gate if memory_efficient else torch.empty_like(gate)
        up_grad = up if memory_efficient else torch.empty_like(up)

        # the kernel uses the gate and up tensors to store the gradients in-place for memory savings
        swiglu_backward_triton_kernel[grid](
            gate_ptr=gate.view(-1),
            up_ptr=up.view(-1),
            output_grad_ptr=output_grad.view(-1),
            gate_grad_ptr=gate_grad.view(-1),
            up_grad_ptr=up_grad.view(-1),
            num_elements=num_elements,
            BLOCK_SIZE=BACKWARD_BLOCK_SIZE,
        )

        return gate_grad, up_grad, None


def swiglu_triton(gate: torch.Tensor, up: torch.Tensor, memory_efficient: bool = False) -> torch.Tensor:
    """swiglu

    Args:
        x (torch.Tensor): input tensor
        y (torch.Tensor): input tensor
        memory_efficient (bool, optional): whether to do an in-place op, uses `gate` and `up` to store gradients in the
            backward pass if set to True. Defaults to False.

    Returns:
        torch.Tensor: output tensor
    """

    return _Swiglu_Triton.apply(gate, up, memory_efficient)
