import torch
import triton

from .torch_implementation import embedding_torch
from .triton_implementation import embedding_forward_triton_kernel


_KERNEL_NAME = "embedding_forward_triton"


class _Embedding_Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_ids: torch.Tensor, wte: torch.Tensor) -> torch.Tensor:
        assert input_ids.is_cuda, "tensor input_ids is not on GPU"
        assert wte.is_cuda, "tensor wte is not on GPU"

        num_tokens = input_ids.numel()
        hidden_size = wte.size(-1)

        output = torch.empty(input_ids.numel(), hidden_size, device=input_ids.device)

        grid = lambda meta: (
            triton.cdiv(num_tokens, meta["BLOCK_SIZE_B"]),
            triton.cdiv(hidden_size, meta["BLOCK_SIZE_H"]),
        )

        with torch.device(input_ids.device):
            embedding_forward_triton_kernel[grid](
                x_ptr=input_ids,
                wte_ptr=wte,
                wte_stride_v=wte.stride(0),
                wte_stride_h=wte.stride(1),
                output_ptr=output,
                output_stride_b=output.stride(0),
                output_stride_h=output.stride(1),
                B=num_tokens,
                H=hidden_size,
                BLOCK_SIZE_B=64,
                BLOCK_SIZE_H=64,
            )

        return output.view(*input_ids.size(), hidden_size)


def embedding_triton(input_ids: torch.Tensor, wte: torch.Tensor) -> torch.Tensor:
    return _Embedding_Triton.apply(input_ids, wte)
