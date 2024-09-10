import torch

from ...kernel_registry import KernelRegistry


_FORWARD_KERNEL_NAME = "swiglu_forward_cuda"
_BACKWARD_KERNEL_NAME = "swiglu_backward_cuda"


class _Swiglu_CUDA(torch.autograd.Function):
    @torch.profiler.record_function(f"khd:{_FORWARD_KERNEL_NAME}")
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(gate, up)

        if not hasattr(_Swiglu_CUDA.forward, "_kernel"):
            _Swiglu_CUDA.forward._kernel = KernelRegistry.get_kernel(_FORWARD_KERNEL_NAME)
            _Swiglu_CUDA.backward._kernel = KernelRegistry.get_kernel(_BACKWARD_KERNEL_NAME)

        return _Swiglu_CUDA.forward._kernel(gate, up)

    @torch.profiler.record_function(f"khd:{_BACKWARD_KERNEL_NAME}")
    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        gate, up = ctx.saved_tensors
        return _Swiglu_CUDA.backward._kernel(gate, up, output_grad)


def swiglu_cuda(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return _Swiglu_CUDA.apply(gate, up)
