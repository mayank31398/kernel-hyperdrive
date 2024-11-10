import torch
import triton

from ...enums import KernelBackend
from .triton_implementation import lightning_transformer_forward_triton_kernel


class _Embedding_KHD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_ids: torch.Tensor, wte: torch.Tensor) -> torch.Tensor:
        batch_size = input_ids.size(0)
        sequence_length = input_ids.size(1)
        hidden_size = wte.size(-1)

        output = torch.empty(batch_size, sequence_length, hidden_size, dtype=wte.dtype, device=input_ids.device)

        grid = lambda meta: (triton.cdiv(batch_size, meta["BLOCK_SIZE_B"]),)

        with torch.device(input_ids.device):
            lightning_transformer_forward_triton_kernel[grid](
                x_ptr=input_ids,
                wte_ptr=wte,
                wte_stride_v=wte.stride(0),
                wte_stride_h=wte.stride(1),
                output_ptr=output,
                output_stride_b=output.stride(0),
                output_stride_h=output.stride(1),
                B=batch_size,
                H=hidden_size,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
            )

        return output


def embedding_khd(
    input_ids: torch.Tensor, wte: torch.Tensor, kernel_backend: KernelBackend, BLOCK_SIZE_B: int, BLOCK_SIZE_H: int
) -> torch.Tensor:
    return _Embedding_KHD.apply(input_ids, wte, kernel_backend, BLOCK_SIZE_B, BLOCK_SIZE_H)
