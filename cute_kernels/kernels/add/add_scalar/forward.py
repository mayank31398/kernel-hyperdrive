import torch

from ....constants import (
    COMMON_CUDA_BLOCK_SIZES_POWERS_OF_2,
    COMMON_TRITON_BLOCK_SIZES_POWERS_OF_2,
    COMMON_VECTOR_INSTRUCTION_WIDTHS,
    MAX_CUDA_BLOCK_SIZE,
    MAX_FP16_BF16_INSTRUCTION_WIDTH,
)
from ....cutotune import CutoTuneConfig, cutotune, get_cartesian_product_cutotune_configs
from ....enums import KernelBackend
from .cuda_implementation import add_scalar_forward_cuda
from .triton_implementation import add_scalar_forward_triton


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
            condition=lambda **kwargs: kwargs["x"].dtype in [torch.float16, torch.bfloat16],
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
        {"kernel_backend": KernelBackend.triton, "vector_instruction_width": None, "BLOCK_SIZE": MAX_CUDA_BLOCK_SIZE}
    ),
    triggers={"x.dtype"},
)
def _forward(
    x: torch.Tensor,
    y: float,
    kernel_backend: KernelBackend,
    vector_instruction_width: int,
    BLOCK_SIZE: int,
) -> torch.Tensor:
    output = torch.empty_like(x)

    if kernel_backend == KernelBackend.cuda:
        assert x.is_cuda, "tensor x is not on GPU"
        add_scalar_forward_cuda(
            x=x, y=y, output=output, vector_instruction_width=vector_instruction_width, BLOCK_SIZE=BLOCK_SIZE
        )
    elif kernel_backend == KernelBackend.triton:
        assert vector_instruction_width is None
        add_scalar_forward_triton(x=x, y=y, output=output, BLOCK_SIZE=BLOCK_SIZE)
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output
