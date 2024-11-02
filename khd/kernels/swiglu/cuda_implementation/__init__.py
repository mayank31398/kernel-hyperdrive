import torch

from ....constants import LIBRARY_NAME
from .ops import (
    _swiglu_backward_cuda,
    _swiglu_backward_cuda_compilable,
    _swiglu_forward_cuda,
    _swiglu_forward_cuda_compilable,
)


_FORWARD_KERNEL_NAME = "swiglu_forward_cuda"
_BACKWARD_KERNEL_NAME = "swiglu_backward_cuda"


class _Swiglu_CUDA(torch.autograd.Function):
    @torch.profiler.record_function(f"{LIBRARY_NAME}:{_FORWARD_KERNEL_NAME}")
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(gate, up)

        BLOCK_SIZE = 1024

        if torch.compiler.is_compiling():
            output = _swiglu_forward_cuda_compilable(gate, up, BLOCK_SIZE)
        else:
            output = _swiglu_forward_cuda(gate, up, BLOCK_SIZE)

        return output

    @torch.profiler.record_function(f"{LIBRARY_NAME}:{_BACKWARD_KERNEL_NAME}")
    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        gate, up = ctx.saved_tensors

        BLOCK_SIZE = 1024

        if torch.compiler.is_compiling():
            gate_grad, up_grad = _swiglu_backward_cuda_compilable(gate, up, output_grad, BLOCK_SIZE)
        else:
            gate_grad, up_grad = _swiglu_backward_cuda(gate, up, output_grad, BLOCK_SIZE)

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
