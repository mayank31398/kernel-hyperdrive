import torch

from ...enums import KernelBackend
from ...utils import (
    CutoTuneConfig,
    CutoTuneParameter,
    ceil_divide,
    cutotune,
    ensure_contiguous,
    get_block_sizes_powers_of_2,
    get_cartesian_product_cutotune_configs,
)
from .torch_implementation import embedding_torch
from .triton_implementation import embedding_backward_triton_kernel, embedding_forward_triton_kernel


class _Embedding_Cute(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    @cutotune(
        configs=get_cartesian_product_cutotune_configs(
            kernel_backend_forward=[KernelBackend.triton],
            BLOCK_SIZE_B_forward=get_block_sizes_powers_of_2(128, 65536),
            BLOCK_SIZE_H_forward=get_block_sizes_powers_of_2(128, 65536),
            condition=lambda **kwargs: 1024
            <= kwargs["BLOCK_SIZE_B_forward"] * kwargs["BLOCK_SIZE_H_forward"]
            <= 65536,
        ),
        default_config=CutoTuneConfig(
            {"kernel_backend_forward": KernelBackend.triton, "BLOCK_SIZE_B_forward": 128, "BLOCK_SIZE_H_forward": 128}
        ),
        triggers={"weight.dtype"},
    )
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
        num_elements = input_ids.numel()
        hidden_size = weight.size(-1)

        output = torch.empty(num_elements, hidden_size, dtype=weight.dtype, device=input_ids.device)

        if kernel_backend_forward == KernelBackend.triton:
            with torch.device(input_ids.device):
                embedding_forward_triton_kernel[
                    (ceil_divide(num_elements, BLOCK_SIZE_B_forward), ceil_divide(hidden_size, BLOCK_SIZE_H_forward))
                ](
                    x_ptr=input_ids,
                    weight_ptr=weight,
                    output_ptr=output,
                    B=num_elements,
                    H=hidden_size,
                    BLOCK_SIZE_B=BLOCK_SIZE_B_forward,
                    BLOCK_SIZE_H=BLOCK_SIZE_H_forward,
                )
        else:
            raise ValueError(f"unexpected kernel_backend_forward ({kernel_backend_forward})")

        output = output.view(*input_ids.size(), hidden_size)

        ctx.save_for_backward(input_ids, weight)
        ctx.kernel_backend_backward = kernel_backend_backward
        ctx.BLOCK_SIZE_B_backward = BLOCK_SIZE_B_backward
        ctx.BLOCK_SIZE_H_backward = BLOCK_SIZE_H_backward

        return output

    @staticmethod
    @ensure_contiguous
    @cutotune(
        configs=get_cartesian_product_cutotune_configs(
            kernel_backend_backward=[KernelBackend.triton],
            BLOCK_SIZE_B_backward=get_block_sizes_powers_of_2(128, 65536),
            BLOCK_SIZE_H_backward=get_block_sizes_powers_of_2(128, 65536),
            condition=lambda **kwargs: 1024
            <= kwargs["BLOCK_SIZE_B_backward"] * kwargs["BLOCK_SIZE_H_backward"]
            <= 65536,
        ),
        default_config=CutoTuneConfig(
            {
                "kernel_backend_backward": KernelBackend.triton,
                "BLOCK_SIZE_B_backward": 128,
                "BLOCK_SIZE_H_backward": 128,
            }
        ),
        triggers={"weight.dtype"},
    )
    def backward(ctx, output_grad: torch.Tensor) -> torch.Tensor:
        input_ids, weight = ctx.saved_tensors
        kernel_backend_backward = ctx.kernel_backend_backward
        BLOCK_SIZE_B_backward = ctx.BLOCK_SIZE_B_backward
        BLOCK_SIZE_H_backward = ctx.BLOCK_SIZE_H_backward

        hidden_size = weight.size(-1)
        num_elements = input_ids.numel()

        weight_grad = torch.zeros_like(weight)

        if kernel_backend_backward == KernelBackend.triton:
            if weight.dtype == torch.bfloat16:
                raise NotImplementedError("bf16 is not supported with triton backend for backward for embeddings")

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
            raise ValueError(f"unexpected kernel_backend_backward ({BLOCK_SIZE_H_backward})")

        return None, weight_grad, *[None] * 6


def embedding_cute(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    kernel_backend_forward: KernelBackend = CutoTuneParameter(),
    kernel_backend_backward: KernelBackend = CutoTuneParameter(),
    BLOCK_SIZE_B_forward: int = CutoTuneParameter(),
    BLOCK_SIZE_B_backward: int = CutoTuneParameter(),
    BLOCK_SIZE_H_forward: int = CutoTuneParameter(),
    BLOCK_SIZE_H_backward: int = CutoTuneParameter(),
) -> torch.Tensor:
    return _Embedding_Cute.apply(
        input_ids,
        weight,
        kernel_backend_forward,
        kernel_backend_backward,
        BLOCK_SIZE_B_forward,
        BLOCK_SIZE_B_backward,
        BLOCK_SIZE_H_forward,
        BLOCK_SIZE_H_backward,
    )
