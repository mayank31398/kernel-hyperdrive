import torch
import triton

from ...enums import KernelBackend
from ...utils import CutoTuneParameter
from .triton_implementation import lightning_transformer_forward_triton_kernel


class _LightningTransformer_KHD(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_ids: torch.Tensor,
        word_embeddings: torch.Tensor,
        kernel_backend: KernelBackend,
        BLOCK_SIZE_B: int,
        BLOCK_SIZE_S: int,
        BLOCK_SIZE_H: int,
    ) -> torch.Tensor:
        batch_size, sequence_length = input_ids.size()
        vocab_size, hidden_size = word_embeddings.size()

        output = torch.empty(
            batch_size, sequence_length, vocab_size, device=torch.cuda.current_device(), dtype=word_embeddings.dtype
        )

        if kernel_backend == KernelBackend.triton:
            grid = lambda meta: (
                triton.cdiv(batch_size, meta["BLOCK_SIZE_B"]),
                triton.cdiv(sequence_length, meta["BLOCK_SIZE_S"]),
                triton.cdiv(hidden_size, meta["BLOCK_SIZE_H"]),
            )

            with torch.device(input_ids.device):
                lightning_transformer_forward_triton_kernel[grid](
                    x_ptr=input_ids,
                    wte_ptr=word_embeddings,
                    wte_stride_v=word_embeddings.stride(0),
                    wte_stride_h=word_embeddings.stride(1),
                    output_ptr=output,
                    output_stride_b=output.stride(0),
                    output_stride_s=output.stride(1),
                    output_stride_h=output.stride(2),
                    B=batch_size,
                    S=sequence_length,
                    H=hidden_size,
                    BLOCK_SIZE_B=BLOCK_SIZE_B,
                    BLOCK_SIZE_S=BLOCK_SIZE_S,
                    BLOCK_SIZE_H=BLOCK_SIZE_H,
                )
        else:
            raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

        return output


def lightning_transformer_khd(
    input_ids: torch.Tensor,
    word_embeddings: torch.Tensor,
    kernel_backend: KernelBackend | CutoTuneParameter = CutoTuneParameter(),
    BLOCK_SIZE_B: int | CutoTuneParameter = CutoTuneParameter(),
    BLOCK_SIZE_S: int | CutoTuneParameter = CutoTuneParameter(),
    BLOCK_SIZE_H: int | CutoTuneParameter = CutoTuneParameter(),
) -> torch.Tensor:
    return _LightningTransformer_KHD.apply(
        input_ids, word_embeddings, kernel_backend, BLOCK_SIZE_B, BLOCK_SIZE_S, BLOCK_SIZE_H
    )
