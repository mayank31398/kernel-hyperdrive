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
    tp,
    sequence_length,
    in_features,
    out_features,
    stride_input_batch_size,
    stride_input_tp,
    stride_input_sequence_length,
    stride_input_in_features,
    stride_weight_tp,
    stride_weight_in_features,
    stride_weight_out_features,
    BLOCK_SIZE_batch_size: tl.constexpr,
    BLOCK_SIZE_tp: tl.constexpr,
    BLOCK_SIZE_sequence_length: tl.constexpr,
    BLOCK_SIZE_out_features: tl.constexpr,
    GROUP_SIZE_sequence_length: tl.constexpr,
):
    # program ids
    pid_batch_size = tl.program_id(axis=0)
    pid_tp = tl.program_id(axis=1)
    pid_matmul = tl.program_id(axis=2)

    # num pids along axes
    num_pid_sequence_length = tl.cdiv(sequence_length, BLOCK_SIZE_sequence_length)
    num_pid_out_features = tl.cdiv(out_features, BLOCK_SIZE_out_features)
    # num pids in a group for maximizing L2 cache hits
    num_pid_in_group = GROUP_SIZE_sequence_length * num_pid_out_features
    # group id
    group_id = pid_matmul // num_pid_in_group

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    block_indices_batch_size = pid_batch_size * BLOCK_SIZE_batch_size + tl.arange(0, BLOCK_SIZE_batch_size)
    block_indices_tp = pid_tp * BLOCK_SIZE_tp + tl.arange(0, BLOCK_SIZE_tp)

    tl.device_print("a", block_indices_batch_size)
    tl.device_print("a", block_indices_tp)

    tl.device_print("a", BLOCK_SIZE_batch_size)
    tl.device_print("a", BLOCK_SIZE_tp)

    # load_mask = block_indices_batch_size < batch_size and block_indices_tp < tp

    # input_block = tl.load()
    # block_start = pid * BLOCK_SIZE
    # block_indices = block_start + tl.arange(0, BLOCK_SIZE)

    # mask = block_indices < num_elements

    # x = tl.load(x_ptr + block_indices, mask=mask)
    # y = tl.load(y_ptr + block_indices, mask=mask)

    # output = x + y

    # tl.store(output_ptr + block_indices, output, mask=mask)


class _EnsembleLinear_Triton(torch.autograd.Function):
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # input -> (batch_size, TP, sequence_length, in_features)
        # weight -> (TP, in_features, out_features)

        assert input.dim() == 4
        assert weight.dim() == 3

        assert input.shape[3] == weight.shape[1]

        batch_size, _, sequence_length, in_features = input.shape
        tp, _, out_features = weight.shape

        if input.shape[1] != tp:
            assert input.shape[1] == 1
            input = input.expand(-1, tp, -1, -1)

        input = input.contiguous()

        output = torch.empty(batch_size, tp, sequence_length, out_features, dtype=input.dtype, device=input.device)

        grid = lambda meta: (
            triton.cdiv(batch_size, meta["BLOCK_SIZE_batch_size"]),
            triton.cdiv(tp, meta["BLOCK_SIZE_tp"]),
            triton.cdiv(sequence_length, meta["BLOCK_SIZE_sequence_length"])
            * triton.cdiv(out_features, meta["BLOCK_SIZE_out_features"]),
        )

        stride_input_batch_size, stride_input_tp, stride_input_sequence_length, stride_input_in_features = (
            input.stride()
        )
        stride_weight_tp, stride_weight_in_features, stride_weight_out_features = weight.stride()

        _ensemble_linear_forward[grid](
            input,
            weight,
            output,
            batch_size,
            tp,
            sequence_length,
            in_features,
            out_features,
            stride_input_batch_size,
            stride_input_tp,
            stride_input_sequence_length,
            stride_input_in_features,
            stride_weight_tp,
            stride_weight_in_features,
            stride_weight_out_features,
            BLOCK_SIZE_batch_size=1024,
            BLOCK_SIZE_tp=1024,
            BLOCK_SIZE_sequence_length=1024,
            BLOCK_SIZE_out_features=1024,
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

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0, std=self.std)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"tensor_parallel_size={self.tensor_parallel_size}, bias={self.bias is not None}"
        )
