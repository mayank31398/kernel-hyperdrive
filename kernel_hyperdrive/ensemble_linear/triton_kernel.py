from typing import Tuple

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _ensemble_linear_forward(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    TP,
    sequence_length,
    in_features,
    out_features,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    block_indices = block_start + tl.arange(0, BLOCK_SIZE)

    mask = block_indices < num_elements

    x = tl.load(x_ptr + block_indices, mask=mask)
    y = tl.load(y_ptr + block_indices, mask=mask)

    output = x + y

    tl.store(output_ptr + block_indices, output, mask=mask)


class _EnsembleLinear_Triton(torch.autograd.Function):
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # input -> (batch_size, TP, sequence_length, in_features)
        # weight -> (TP, in_features, out_features)

        assert input.dim() == 4
        assert weight.dim() == 3

        assert input.shape[1] == weight.shape[0]
        assert input.shape[3] == weight.shape[1]

        input = input.contiguous()

        batch_size, tp, sequence_length, in_features = input.shape
        out_features = weight.shape[-1]

        output = torch.empty(batch_size, tp, sequence_length, out_features, dtype=input.dtype, device=input.device)

        num_elements = x.numel()
        grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

        _ensemble_linear_forward[grid](
            input,
            weight,
            output,
            batch_size,
            tp,
            sequence_length,
            in_features,
            out_features,
            BLOCK_SIZE_M=1024,
            BLOCK_SIZE_N=1024,
        )

        return output

    def backward(ctx, output_grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return output_grad, output_grad


def ensemble_linear_triton(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _EnsembleLinear_Triton.apply(input, weight)


class EnsembleLinear_Triton(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, tensor_parallel_size: int, std: float, bias: bool = True
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.tensor_parallel_size = tensor_parallel_size
        self.std = std

        self.weight = nn.Parameter(torch.empty(self.tensor_parallel_size, self.in_features, self.out_features))

        if bias:
            raise NotImplementedError(f"bias is not supported with {self.__class__.__name__}")

        self.reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return ensemble_linear_triton(input, self.weight)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0, std=self.std)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"tensor_parallel_size={self.tensor_parallel_size}, bias={self.bias is not None}"
        )
