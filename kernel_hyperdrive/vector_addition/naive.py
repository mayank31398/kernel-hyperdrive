from typing import Tuple

import torch
import torch.nn as nn


def _vector_addition_naive(
    x: torch.Tensor, y: torch.Tensor, output: torch.Tensor, num_elements: int, BLOCK_SIZE: int
) -> None:
    for block_start in range(0, num_elements, BLOCK_SIZE):
        block_end = max(block_start + BLOCK_SIZE, num_elements)

        output[block_start:block_end] = x[block_start:block_end] + y[block_start:block_end]


class _VectorAddition_Naive(torch.autograd.Function):
    def forward(ctx, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 1
        output = torch.empty_like(x)

        num_elements = x.numel()

        _vector_addition_naive(x, y, num_elements, BLOCK_SIZE=1024)

        return output

    def backward(ctx, output_grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return output_grad, output_grad


def vector_addition_naive(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _VectorAddition_Naive.apply(x, y)


class VectorAddition_Naive(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return vector_addition_naive(x, y)
