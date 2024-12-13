import torch

from ....constants import (
    COMMON_TRITON_BLOCK_SIZES_POWERS_OF_2,
    LIBRARY_NAME,
    MAX_TRITON_BLOCK_SIZE,
    TORCH_TO_TRITON_DTYPE,
)
from ....utils import CutoTuneConfig, ceil_divide, cute_op, cutotune, get_powers_of_2
from .kernels_backward import rmsnorm_backward_triton_kernel
from .kernels_forward import rmsnorm_forward_triton_kernel


_FORWARD_KERNEL_NAME = "rmsnorm_forward_triton"
_BACKWARD_KERNEL_NAME = "rmsnorm_backward_triton"


@cutotune(
    configs=[
        CutoTuneConfig(
            {"BLOCK_SIZE_B": BLOCK_SIZE_B},
            condition=lambda **kwargs: 1024
            <= kwargs["BLOCK_SIZE_B"] * kwargs["BLOCK_SIZE_H"]
            <= MAX_TRITON_BLOCK_SIZE,
        )
        for BLOCK_SIZE_B in get_powers_of_2(1, 32) + COMMON_TRITON_BLOCK_SIZES_POWERS_OF_2
    ],
    default_config=CutoTuneConfig({"BLOCK_SIZE_B": 1}),
    triggers={"x.dtype", "BLOCK_SIZE_H"},
)
@cute_op(f"{LIBRARY_NAME}::{_FORWARD_KERNEL_NAME}", mutates_args={"output", "rmsnorm_denominator"})
def rmsnorm_forward_triton(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    output: torch.Tensor,
    eps: float,
    memory_efficient: bool,
    rmsnorm_denominator: torch.Tensor | None,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> None:
    num_elements, hidden_size = x.size()

    if BLOCK_SIZE_H < hidden_size:
        raise ValueError(f"hidden_size should be more than the BLOCK_SIZE_H")

    with torch.device(x.device):
        rmsnorm_forward_triton_kernel[(ceil_divide(num_elements, BLOCK_SIZE_B),)](
            x_ptr=x,
            x_dtype=TORCH_TO_TRITON_DTYPE[x.dtype],
            has_weight=weight is not None,
            weight_ptr=weight,
            output_ptr=output,
            eps=eps,
            memory_efficient=memory_efficient,
            rmsnorm_denominator_ptr=rmsnorm_denominator,
            B=num_elements,
            H=hidden_size,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
