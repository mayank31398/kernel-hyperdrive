import torch
import triton

from .kernels import add_tensor_forward_triton_kernel


class _AddTensor_Triton(torch.autograd.Function):
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

        with torch.device(x.device):
            add_tensor_forward_triton_kernel[grid](
                x_ptr=x, y_ptr=y, output_ptr=output, num_elements=num_elements, BLOCK_SIZE=BLOCK_SIZE
            )

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor]:
        return output_grad, output_grad


def add_tensor_triton(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """tensor addition

    Args:
        x (torch.Tensor): input tensor
        y (torch.Tensor): input tensor

    Returns:
        torch.Tensor: output tensor
    """

    return _AddTensor_Triton.apply(x, y)
