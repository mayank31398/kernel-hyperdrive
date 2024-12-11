import torch
import triton

from ...constants import COMMON_TRITON_BLOCK_SIZES_POWERS_OF_2, MAX_TRITON_BLOCK_SIZE, TORCH_TO_TRITON_DTYPE
from ...enums import KernelBackend
from ...utils import CutoTuneConfig, ceil_divide, cutotune
from .triton_implementation import rmsnorm_forward_triton_kernel


@cutotune(
    configs=[
        CutoTuneConfig(
            {"BLOCK_SIZE_B": BLOCK_SIZE_B},
            condition=lambda **kwargs: 1024
            <= kwargs["BLOCK_SIZE_B"] * kwargs["BLOCK_SIZE_H"]
            <= MAX_TRITON_BLOCK_SIZE,
        )
        for BLOCK_SIZE_B in [1, 2, 4, 8, 16, 32] + COMMON_TRITON_BLOCK_SIZES_POWERS_OF_2
    ],
    default_config=CutoTuneConfig({"BLOCK_SIZE_B": 1}),
    triggers={"x.dtype", "BLOCK_SIZE_H"},
)
def _triton_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    rmsnorm_denominator: torch.Tensor,
    eps: float,
    memory_efficient: bool,
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


@cutotune(
    configs=[CutoTuneConfig({"kernel_backend": KernelBackend.triton})],
    default_config=CutoTuneConfig({"kernel_backend": KernelBackend.triton}),
    triggers={"x.dtype"},
)
def _forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    memory_efficient: bool,
    kernel_backend: KernelBackend,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> tuple[torch.Tensor | None]:
    hidden_size = x.size(-1)
    num_elements = x.numel() // hidden_size

    output = torch.empty_like(x)
    rmsnorm_denominator = None if memory_efficient else torch.empty(num_elements, device=x.device, dtype=torch.float32)

    if kernel_backend == KernelBackend.triton:
        BLOCK_SIZE_H = triton.next_power_of_2(hidden_size)
        assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE

        _triton_forward(
            x=x.view(-1, hidden_size),
            weight=weight,
            output=output.view(-1, hidden_size),
            rmsnorm_denominator=rmsnorm_denominator,
            eps=eps,
            memory_efficient=memory_efficient,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output, rmsnorm_denominator
