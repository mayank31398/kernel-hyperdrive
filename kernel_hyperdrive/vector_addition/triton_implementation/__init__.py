import torch
import triton

from .kernel import vector_addition_forward_triton_kernel


class _VectorAddition_Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 1

        assert x.is_cuda
        assert y.is_cuda

        assert x.is_contiguous()
        assert y.is_contiguous()

        output = torch.empty_like(x)

        num_elements = x.numel()
        grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

        vector_addition_forward_triton_kernel[grid](x, y, output, num_elements, BLOCK_SIZE=1024)

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return output_grad, output_grad


def vector_addition_triton(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _VectorAddition_Triton.apply(x, y)
