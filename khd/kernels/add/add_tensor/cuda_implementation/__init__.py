import torch

from .ops import _add_tensor_forward_cuda, _add_tensor_forward_cuda_compilable


class _AddTensor_CUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if torch.compiler.is_compiling():
            output = _add_tensor_forward_cuda_compilable(x, y)
        else:
            output = _add_tensor_forward_cuda(x, y)

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor]:
        return output_grad, output_grad


def add_tensor_cuda(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """tensor addition

    Args:
        x (torch.Tensor): input tensor
        y (torch.Tensor): input tensor

    Returns:
        torch.Tensor: output tensor
    """

    return _AddTensor_CUDA.apply(x, y)
