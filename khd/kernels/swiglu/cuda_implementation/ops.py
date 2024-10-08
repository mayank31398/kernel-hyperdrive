import torch

from ....constants import LIBRARY_NAME
from ....kernel_registry import KernelRegistry
from ...utils import torch_custom_op


_FORWARD_KERNEL_NAME = "swiglu_forward_cuda"
_BACKWARD_KERNEL_NAME = "swiglu_backward_cuda"


@torch_custom_op(f"{LIBRARY_NAME}::{_FORWARD_KERNEL_NAME}", mutates_args={})
def _swiglu_forward_cuda_compilable(gate: torch.Tensor, up: torch.Tensor, BLOCK_SIZE: int) -> torch.Tensor:
    return KernelRegistry.get_kernel(_FORWARD_KERNEL_NAME)(gate, up, BLOCK_SIZE)


@_swiglu_forward_cuda_compilable.register_fake
def _(gate: torch.Tensor, up: torch.Tensor, BLOCK_SIZE: int) -> torch.Tensor:
    return torch.empty_like(gate)


@torch_custom_op(f"{LIBRARY_NAME}::{_BACKWARD_KERNEL_NAME}", mutates_args={})
def _swiglu_backward_cuda_compilable(
    gate: torch.Tensor, up: torch.Tensor, output_grad: torch.Tensor, BLOCK_SIZE: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return KernelRegistry.get_kernel(_BACKWARD_KERNEL_NAME)(gate, up, output_grad, BLOCK_SIZE)


@_swiglu_backward_cuda_compilable.register_fake
def _(
    gate: torch.Tensor, up: torch.Tensor, output_grad: torch.Tensor, BLOCK_SIZE: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(gate), torch.empty_like(gate)


class _Swiglu_CUDA(torch.autograd.Function):
    @torch.profiler.record_function(f"{LIBRARY_NAME}:{_FORWARD_KERNEL_NAME}")
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(gate, up)

        BLOCK_SIZE = 1024

        if torch.compiler.is_compiling():
            output = _swiglu_forward_cuda_compilable(gate, up, BLOCK_SIZE)
        else:
            kernel = KernelRegistry.get_kernel(_FORWARD_KERNEL_NAME)
            output = kernel(gate, up, BLOCK_SIZE)

        return output

    @torch.profiler.record_function(f"{LIBRARY_NAME}:{_BACKWARD_KERNEL_NAME}")
    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        gate, up = ctx.saved_tensors

        BLOCK_SIZE = 1024

        if torch.compiler.is_compiling():
            gate_grad, up_grad = _swiglu_backward_cuda_compilable(gate, up, BLOCK_SIZE)
        else:
            kernel = KernelRegistry.get_kernel(_BACKWARD_KERNEL_NAME)
            gate_grad, up_grad = kernel(gate, up, output_grad, BLOCK_SIZE)

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
