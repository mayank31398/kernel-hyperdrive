import torch

from ...constants import (
    COMMON_CUDA_BLOCK_SIZES_POWERS_OF_2,
    COMMON_TRITON_BLOCK_SIZES_POWERS_OF_2,
    COMMON_VECTOR_INSTRUCTION_WIDTHS,
    MAX_FP16_BF16_INSTRUCTION_WIDTH,
)
from ...cutotune import CutoTuneConfig, cutotune, get_cartesian_product_cutotune_configs
from ...enums import KernelBackend
from .cuda_implementation import swiglu_backward_cuda
from .triton_implementation import swiglu_backward_triton


@cutotune(
    configs=(
        get_cartesian_product_cutotune_configs(
            kernel_backend=[KernelBackend.cuda],
            vector_instruction_width=COMMON_VECTOR_INSTRUCTION_WIDTHS,
            BLOCK_SIZE=COMMON_CUDA_BLOCK_SIZES_POWERS_OF_2,
        )
        if torch.cuda.is_available()
        else []
    )
    + (
        get_cartesian_product_cutotune_configs(
            kernel_backend=[KernelBackend.cuda],
            vector_instruction_width=[MAX_FP16_BF16_INSTRUCTION_WIDTH],
            BLOCK_SIZE=COMMON_CUDA_BLOCK_SIZES_POWERS_OF_2,
            condition=lambda **kwargs: kwargs["gate"].dtype in [torch.float16, torch.bfloat16],
        )
        if torch.cuda.is_available()
        else []
    )
    + get_cartesian_product_cutotune_configs(
        kernel_backend=[KernelBackend.triton],
        vector_instruction_width=[None],
        BLOCK_SIZE=COMMON_TRITON_BLOCK_SIZES_POWERS_OF_2,
    ),
    default_config=CutoTuneConfig(
        {"kernel_backend": KernelBackend.triton, "vector_instruction_width": None, "BLOCK_SIZE": 1024}
    ),
    triggers={"gate.dtype"},
)
def _backward(
    gate: torch.Tensor,
    up: torch.Tensor,
    output_grad: torch.Tensor,
    kernel_backend: KernelBackend,
    vector_instruction_width: int,
    BLOCK_SIZE: int,
) -> tuple[torch.Tensor]:
    gate_grad = torch.empty_like(gate)
    up_grad = torch.empty_like(up)

    if kernel_backend == KernelBackend.cuda:
        swiglu_backward_cuda(
            gate=gate,
            up=up,
            output_grad=output_grad,
            gate_grad=gate_grad,
            up_grad=up_grad,
            vector_instruction_width=vector_instruction_width,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    elif kernel_backend == KernelBackend.triton:
        assert vector_instruction_width is None
        swiglu_backward_triton(
            gate=gate, up=up, output_grad=output_grad, gate_grad=gate_grad, up_grad=up_grad, BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return gate_grad, up_grad
