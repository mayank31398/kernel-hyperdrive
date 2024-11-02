import torch

from .ops import _add_scalar_forward_cuda, _add_scalar_forward_cuda_compilable


class _AddScalar_CUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: float, BLOCK_SIZE: int) -> torch.Tensor:
        if y == 0:
            return x

        output = torch.empty_like(x)

        if torch.compiler.is_compiling():
            _add_scalar_forward_cuda_compilable(x=x, y=y, output=output, BLOCK_SIZE=BLOCK_SIZE)
        else:
            _add_scalar_forward_cuda(x=x, y=y, output=output, BLOCK_SIZE=BLOCK_SIZE)

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        return output_grad, None, None


def add_scalar_cuda(x: torch.Tensor, y: float, BLOCK_SIZE: int) -> torch.Tensor:
    """tensor addition

    Args:
        x (torch.Tensor): input tensor
        y (float): input scalar
        BLOCK_SIZE (int): block size

    Returns:
        torch.Tensor: output tensor
    """

    return _AddScalar_CUDA.apply(x, y, BLOCK_SIZE)
