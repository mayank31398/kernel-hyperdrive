import torch

from ....constants import (
    COMMON_CUDA_BLOCK_SIZES_POWERS_OF_2,
    COMMON_TRITON_BLOCK_SIZES_POWERS_OF_2,
    COMMON_VECTOR_INSTRUCTION_WIDTHS,
    MAX_CUDA_BLOCK_SIZE,
    MAX_FP16_BF16_INSTRUCTION_WIDTH,
)
from ....enums import KernelBackend
from ....utils import CutoTuneConfig, ceil_divide, cutotune, get_cartesian_product_cutotune_configs
from .cuda_implementation import add_scalar_forward_cuda_kernel
from .triton_implementation import add_scalar_forward_triton_kernel


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

        add_scalar_forward_cuda_kernel(
            x=x, y=y, output=output, vector_instruction_width=vector_instruction_width, BLOCK_SIZE=BLOCK_SIZE
        )
    elif kernel_backend == KernelBackend.triton:
        assert vector_instruction_width is None

        num_elements = x.numel()
        num_programs = ceil_divide(num_elements, BLOCK_SIZE)

        with torch.device(x.device):
            add_scalar_forward_triton_kernel[num_programs,](
                x_ptr=x, y=y, output_ptr=output, num_elements=num_elements, BLOCK_SIZE=BLOCK_SIZE
            )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output
