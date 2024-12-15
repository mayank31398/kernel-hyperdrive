import torch

from ...cutotune import CutoTuneConfig, cutotune, get_cartesian_product_cutotune_configs
from ...enums import KernelBackend
from ...math import get_powers_of_2
from .triton_implementation import swiglu_unchunked_backward_triton


@cutotune(
    configs=get_cartesian_product_cutotune_configs(
        kernel_backend=[KernelBackend.triton],
        BLOCK_SIZE_B=get_powers_of_2(64, 1024),
        BLOCK_SIZE_H=[64],
    ),
    default_config=CutoTuneConfig({"kernel_backend": KernelBackend.triton, "BLOCK_SIZE_B": 64, "BLOCK_SIZE_H": 64}),
    triggers={"x.dtype"},
)
def _backward(
    x: torch.Tensor,
    output_grad: torch.Tensor,
    kernel_backend: KernelBackend,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> tuple[torch.Tensor]:
    x_grad = torch.empty_like(x)

    if kernel_backend == KernelBackend.triton:
        swiglu_unchunked_backward_triton(
            x=x, output_grad=output_grad, x_grad=x_grad, BLOCK_SIZE_B=BLOCK_SIZE_B, BLOCK_SIZE_H=BLOCK_SIZE_H
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return x_grad
