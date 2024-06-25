from typing import Tuple

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _vector_addition_forward(x_ptr, y_ptr, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    block_indices = block_start + tl.arange(0, BLOCK_SIZE)

    mask = block_indices < num_elements

    x = tl.load(x_ptr + block_indices, mask=mask)
    y = tl.load(y_ptr + block_indices, mask=mask)

    output = x + y

    tl.store(output_ptr + block_indices, output, mask=mask)


class _VectorAddition_Triton(torch.autograd.Function):
    def forward(ctx, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 1
        output = torch.empty_like(x)

        num_elements = x.numel()
        grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

        _vector_addition_forward[grid](x, y, output, num_elements, BLOCK_SIZE=1024)

        return output

    def backward(ctx, output_grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return output_grad, output_grad


def vector_addition_triton(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _VectorAddition_Triton.apply(x, y)


class VectorAddition_Triton(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return vector_addition_triton(x, y)
