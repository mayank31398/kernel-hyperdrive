import torch
import triton

from .kernels import add_scalar_forward_triton_kernel


class _AddScalar_Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: float) -> torch.Tensor:
        assert x.is_cuda, "tensor x is not on GPU"

        if y == 0:
            return x

        output = torch.empty_like(x)

        num_elements = x.numel()
        grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

        BLOCK_SIZE = 1024

        add_scalar_forward_triton_kernel[grid](
            x.view(-1), y.view(-1), output.view(-1), num_elements, BLOCK_SIZE=BLOCK_SIZE
        )

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        return output_grad, None


def add_scalar_triton(x: torch.Tensor, y: float) -> torch.Tensor:
    """vector addition

    Args:
        x (torch.Tensor): input tensor
        y (float): input scalar

    Returns:
        torch.Tensor: output tensor
    """

    return _AddScalar_Triton.apply(x, y)
