import torch

from ...enums import KernelBackend
from ...utils import ceil_divide, ensure_contiguous
from .forward import _forward
from .torch_implementation import embedding_torch
from .triton_implementation import embedding_backward_triton_kernel, embedding_forward_triton_kernel


class _Embedding_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        input_ids: torch.Tensor,
        weight: torch.Tensor,
        kernel_backend_forward: KernelBackend,
        kernel_backend_backward: KernelBackend,
        BLOCK_SIZE_B_forward: int,
        BLOCK_SIZE_B_backward: int,
        BLOCK_SIZE_H_forward: int,
        BLOCK_SIZE_H_backward: int,
    ) -> torch.Tensor:
        output = _forward(
            input_ids=input_ids,
            weight=weight,
            kernel_backend=kernel_backend_forward,
            BLOCK_SIZE_B=BLOCK_SIZE_B_forward,
            BLOCK_SIZE_H=BLOCK_SIZE_H_forward,
        )

        ctx.save_for_backward(input_ids, weight)
        ctx.kernel_backend_backward = kernel_backend_backward
        ctx.BLOCK_SIZE_B_backward = BLOCK_SIZE_B_backward
        ctx.BLOCK_SIZE_H_backward = BLOCK_SIZE_H_backward

        return output

    @staticmethod
    @ensure_contiguous
    def forward(ctx, output_grad: torch.Tensor) -> torch.Tensor:
        input_ids, weight = ctx.saved_tensors
        kernel_backend_backward = ctx.kernel_backend_backward
        BLOCK_SIZE_B_backward = ctx.BLOCK_SIZE_B_backward
        BLOCK_SIZE_H_backward = ctx.BLOCK_SIZE_H_backward

        hidden_size = weight.size(-1)
        num_elements = input_ids.numel()

        weight_grad = torch.zeros_like(weight)

        if kernel_backend_backward == KernelBackend.triton:
            with torch.device(input_ids.device):
                embedding_backward_triton_kernel[
                    (ceil_divide(num_elements, BLOCK_SIZE_B_backward), ceil_divide(hidden_size, BLOCK_SIZE_H_backward))
                ](
                    x_ptr=input_ids,
                    output_grad_ptr=output_grad,
                    weight_grad_ptr=weight_grad,
                    B=num_elements,
                    H=hidden_size,
                    BLOCK_SIZE_B=BLOCK_SIZE_B_backward,
                    BLOCK_SIZE_H=BLOCK_SIZE_H_backward,
                )
        else:
            raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

        return None, weight_grad, *[None] * 6


def embedding_cute(
    input_ids: torch.Tensor, weight: torch.Tensor, kernel_backend: KernelBackend, BLOCK_SIZE_B: int, BLOCK_SIZE_H: int
) -> torch.Tensor:
    return _Embedding_Cute.apply(input_ids, weight, kernel_backend, BLOCK_SIZE_B, BLOCK_SIZE_H)
