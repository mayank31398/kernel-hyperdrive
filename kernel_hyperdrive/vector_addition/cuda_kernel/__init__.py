from typing import Tuple

import torch
import torch.nn as nn


class _VectorAddition_CUDA(torch.autograd.Function):
    def forward(ctx, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        import vector_addition_cuda

        return vector_addition_cuda.vector_addition_forward(x, y)

    def backward(ctx, output_grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return output_grad, output_grad


def vector_addition_cuda(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _VectorAddition_CUDA.apply(x, y)


class VectorAddition_CUDA(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return vector_addition_cuda(x, y)
