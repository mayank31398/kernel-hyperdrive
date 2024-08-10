import torch
import torch.nn as nn


class _VectorAddition_CUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        import khd

        return khd.vector_addition_forward(x, y)

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return output_grad, output_grad


# this registers the kernel with PyTorch to make it work with torch.compile
@torch.library.custom_op("khd::vector_addition_cuda", mutates_args=())
def vector_addition_cuda(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _VectorAddition_CUDA.apply(x, y)


# this tells torch.compile the output shape given the input shape
@vector_addition_cuda.register_fake
def _(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


class VectorAddition_CUDA(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return vector_addition_cuda(x, y)
