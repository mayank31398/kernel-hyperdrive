import torch
import triton

from .kernels import vector_addition_forward_triton_kernel


_KERNEL_NAME = "vector_addition_forward_triton"


class _VectorAddition_Triton(torch.autograd.Function):
    @torch.profiler.record_function(_KERNEL_NAME)
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "tensor x is not on GPU"
        assert y.is_cuda, "tensor y is not on GPU"

        assert x.size() == y.size(), "tensors x and y should have same shape"
        assert x.type() == y.type(), "tensors x and y should have same dtype"

        output = torch.empty_like(x)

        original_shape = x.size()
        ctx.original_shape = original_shape

        x = x.view(-1)
        y = y.view(-1)

        num_elements = x.numel()
        grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

        vector_addition_forward_triton_kernel[grid](x, y, output, num_elements, BLOCK_SIZE=1024)

        output = output.view(original_shape)

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output_grad = output_grad.view(ctx.original_shape)
        return output_grad, output_grad


def vector_addition_triton(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _VectorAddition_Triton.apply(x, y)
