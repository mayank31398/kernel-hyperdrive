import torch

from .ops import _add_tensor_forward_cuda, _add_tensor_forward_cuda_compilable


class _AddTensor_CUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor, vectorized_loop_size: int) -> torch.Tensor:
        BLOCK_SIZE = 1024

        if torch.compiler.is_compiling():
            output = _add_tensor_forward_cuda_compilable(
                x=x, y=y, vectorized_loop_size=vectorized_loop_size, BLOCK_SIZE=BLOCK_SIZE
            )
        else:
            output = _add_tensor_forward_cuda(x, y, vectorized_loop_size=vectorized_loop_size, BLOCK_SIZE=BLOCK_SIZE)

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor]:
        return output_grad, output_grad, None


def add_tensor_cuda(x: torch.Tensor, y: torch.Tensor, vectorized_loop_size: int) -> torch.Tensor:
    """tensor addition

    Args:
        x (torch.Tensor): input tensor
        y (torch.Tensor): input tensor
        vectorized_loop_size (int): vector instructions' operand size

    Returns:
        torch.Tensor: output tensor
    """

    return _AddTensor_CUDA.apply(x, y, vectorized_loop_size)
