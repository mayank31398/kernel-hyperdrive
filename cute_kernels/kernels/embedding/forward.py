import torch

from ...constants import MAX_TRITON_BLOCK_SIZE
from ...enums import KernelBackend
from ...utils import CutoTuneConfig, ceil_divide, cutotune, get_cartesian_product_cutotune_configs, get_powers_of_2
from .triton_implementation import embedding_forward_triton_kernel


@cutotune(
    configs=get_cartesian_product_cutotune_configs(
        kernel_backend=[KernelBackend.triton],
        BLOCK_SIZE_B=get_powers_of_2(128, MAX_TRITON_BLOCK_SIZE),
        BLOCK_SIZE_H=get_powers_of_2(128, MAX_TRITON_BLOCK_SIZE),
        condition=lambda **kwargs: 1024 <= kwargs["BLOCK_SIZE_B"] * kwargs["BLOCK_SIZE_H"] <= MAX_TRITON_BLOCK_SIZE,
    ),
    default_config=CutoTuneConfig({"kernel_backend": KernelBackend.triton, "BLOCK_SIZE_B": 128, "BLOCK_SIZE_H": 128}),
    triggers={"weight.dtype"},
)
def _forward(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    kernel_backend: KernelBackend,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> torch.Tensor:
    num_elements = input_ids.numel()
    hidden_size = weight.size(-1)

    output = torch.empty(num_elements, hidden_size, dtype=weight.dtype, device=input_ids.device)

    if kernel_backend == KernelBackend.triton:
        with torch.device(input_ids.device):
            embedding_forward_triton_kernel[
                (ceil_divide(num_elements, BLOCK_SIZE_B), ceil_divide(hidden_size, BLOCK_SIZE_H))
            ](
                x_ptr=input_ids,
                weight_ptr=weight,
                output_ptr=output,
                B=num_elements,
                H=hidden_size,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
            )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output.view(*input_ids.size(), hidden_size)
