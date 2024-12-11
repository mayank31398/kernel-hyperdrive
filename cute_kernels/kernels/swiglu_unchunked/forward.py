import torch

from ...enums import KernelBackend
from ...utils import (
    CutoTuneConfig,
    ceil_divide,
    cutotune,
    divide_if_divisible,
    get_cartesian_product_cutotune_configs,
    get_powers_of_2,
)
from .triton_implementation import swiglu_unchunked_forward_triton_kernel


@cutotune(
    configs=get_cartesian_product_cutotune_configs(
        kernel_backend=[KernelBackend.triton],
        BLOCK_SIZE_B=get_powers_of_2(64, 1024),
        BLOCK_SIZE_H=[64],
    ),
    default_config=CutoTuneConfig({"kernel_backend": KernelBackend.triton, "BLOCK_SIZE_B": 64, "BLOCK_SIZE_H": 64}),
    triggers={"x.dtype"},
)
def _forward(x: torch.Tensor, kernel_backend: KernelBackend, BLOCK_SIZE_B: int, BLOCK_SIZE_H: int) -> torch.Tensor:
    H = x.size(-1)
    B = x.numel() // H

    output = torch.empty(*x.size()[:-1], divide_if_divisible(H, 2), device=x.device, dtype=x.dtype)

    if kernel_backend == KernelBackend.triton:
        with torch.device(x.device):
            swiglu_unchunked_forward_triton_kernel[(ceil_divide(B, BLOCK_SIZE_B), ceil_divide(H, BLOCK_SIZE_H))](
                x_ptr=x,
                output_ptr=output,
                B=B,
                H=H,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
            )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output
