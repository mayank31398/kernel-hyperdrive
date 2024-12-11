import torch

from ...constants import COMMON_TRITON_BLOCK_SIZES_POWERS_OF_2
from ...enums import KernelBackend
from ...utils import CutoTuneConfig, ceil_divide, cutotune, get_cartesian_product_cutotune_configs
from .triton_implementation import swiglu_unchunked_backward_triton_kernel


@cutotune(
    configs=get_cartesian_product_cutotune_configs(
        kernel_backend=[KernelBackend.triton], BLOCK_SIZE=COMMON_TRITON_BLOCK_SIZES_POWERS_OF_2
    ),
    default_config=CutoTuneConfig({"kernel_backend": KernelBackend.triton, "BLOCK_SIZE": 1024}),
    triggers={"gate.dtype"},
)
def _backward(
    gate: torch.Tensor,
    up: torch.Tensor,
    output_grad: torch.Tensor,
    kernel_backend: KernelBackend,
    BLOCK_SIZE: int,
) -> tuple[torch.Tensor]:
    gate_grad = torch.empty_like(gate)
    up_grad = torch.empty_like(up)

    if kernel_backend == KernelBackend.triton:
        num_elements = gate.numel()

        with torch.device(gate.device):
            swiglu_unchunked_backward_triton_kernel[ceil_divide(num_elements, BLOCK_SIZE),](
                gate_ptr=gate,
                up_ptr=up,
                output_grad_ptr=output_grad,
                gate_grad_ptr=gate_grad,
                up_grad_ptr=up_grad,
                num_elements=num_elements,
                BLOCK_SIZE=BLOCK_SIZE,
            )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return gate_grad, up_grad
