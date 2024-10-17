import torch
import triton

from ....constants import LIBRARY_NAME
from .kernels import lightning_transformer_triton_kernel


_KERNEL_NAME = "embedding_forward_triton"


class _Embedding_Triton(torch.autograd.Function):
    @torch.profiler.record_function(f"{LIBRARY_NAME}:{_KERNEL_NAME}")
    @staticmethod
    def forward(ctx, input_ids: torch.Tensor, wte: torch.Tensor) -> torch.Tensor:
        assert input_ids.is_cuda, "tensor input_ids is not on GPU"
        assert wte.is_cuda, "tensor wte is not on GPU"

        batch_size, sequence_length = input_ids.size()
        hidden_size = wte.size(-1)

        output = torch.empty(batch_size, sequence_length, hidden_size, device=input_ids.device)

        grid = lambda meta: (
            triton.cdiv(batch_size, meta["BLOCK_SIZE_B"]),
            triton.cdiv(sequence_length, meta["BLOCK_SIZE_S"]),
            triton.cdiv(hidden_size, meta["BLOCK_SIZE_H"]),
        )

        lightning_transformer_triton_kernel[grid](
            x_ptr=input_ids,
            x_stride_b=input_ids.stride(0),
            x_stride_s=input_ids.stride(1),
            wte_ptr=wte,
            wte_stride_v=wte.stride(0),
            wte_stride_h=wte.stride(1),
            logits_ptr=output,
            logits_stride_b=output.stride(0),
            logits_stride_s=output.stride(1),
            logits_stride_h=output.stride(2),
            B=batch_size,
            S=sequence_length,
            H=hidden_size,
            BLOCK_SIZE_B=64,
            BLOCK_SIZE_S=64,
            BLOCK_SIZE_H=64,
        )

        return output


def embedding_triton(input_ids: torch.Tensor, wte: torch.Tensor) -> torch.Tensor:
    return _Embedding_Triton.apply(input_ids, wte)
