import torch
import triton

from .kernels import vector_addition_forward_triton_kernel


class _VectorAddition_Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "tensor x is not on GPU"
        assert y.is_cuda, "tensor y is not on GPU"

        assert x.size() == y.size(), "tensors x and y should have same shape"
        assert x.type() == y.type(), "tensors x and y should have same dtype"

        output = torch.empty_like(x)

        num_elements = x.numel()
        grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

        BLOCK_SIZE = 1024

        vector_addition_forward_triton_kernel[grid](
            x.view(-1), y.view(-1), output.view(-1), num_elements, BLOCK_SIZE=BLOCK_SIZE
        )

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        return output_grad, output_grad, None


def vector_addition_triton(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """vector addition

    Args:
        x (torch.Tensor): input tensor
        y (torch.Tensor): input tensor

    Returns:
        torch.Tensor: output tensor
    """

    return _VectorAddition_Triton.apply(x, y)
