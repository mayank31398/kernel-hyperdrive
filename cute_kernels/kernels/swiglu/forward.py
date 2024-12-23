import torch

from ...constants import (
    COMMON_CUDA_BLOCK_SIZES_POWERS_OF_2,
    COMMON_TRITON_BLOCK_SIZES_POWERS_OF_2,
    COMMON_VECTOR_INSTRUCTION_WIDTHS,
    MAX_FP16_BF16_INSTRUCTION_WIDTH,
)
from ...cutotune import CutoTuneConfig, cutotune, get_cartesian_product_cutotune_configs
from ...enums import KernelBackend
from .cuda_implementation import swiglu_forward_cuda
from .triton_implementation import swiglu_forward_triton


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
def _forward(
    gate: torch.Tensor, up: torch.Tensor, kernel_backend: KernelBackend, vector_instruction_width: int, BLOCK_SIZE: int
) -> torch.Tensor:
    output = torch.empty_like(gate)

    if kernel_backend == KernelBackend.cuda:
        assert gate.is_cuda, "tensor gate is not on GPU"
        assert up.is_cuda, "tensor up is not on GPU"

        swiglu_forward_cuda(gate=gate, up=up, output=output, BLOCK_SIZE=BLOCK_SIZE)
    elif kernel_backend == KernelBackend.triton:
        swiglu_forward_triton(gate=gate, up=up, output=output, BLOCK_SIZE=BLOCK_SIZE)
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output
