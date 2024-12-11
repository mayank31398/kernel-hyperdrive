import torch

from ...constants import COMMON_TRITON_BLOCK_SIZES_POWERS_OF_2
from ...enums import KernelBackend
from ...utils import CutoTuneConfig, ceil_divide, cutotune, divide_if_divisible, get_cartesian_product_cutotune_configs
from .triton_implementation import swiglu_unchunked_backward_triton_kernel


@cutotune(
    configs=get_cartesian_product_cutotune_configs(
        kernel_backend=[KernelBackend.triton], BLOCK_SIZE=COMMON_TRITON_BLOCK_SIZES_POWERS_OF_2
    ),
    default_config=CutoTuneConfig({"kernel_backend": KernelBackend.triton, "BLOCK_SIZE": 1024}),
    triggers={"x.dtype"},
)
def _backward(
    x: torch.Tensor,
    output_grad: torch.Tensor,
    kernel_backend: KernelBackend,
    BLOCK_SIZE: int,
) -> tuple[torch.Tensor]:
    x_size = x.size()
    stride = divide_if_divisible(x_size[-1], 2)
    x_grad = torch.empty_like(x)

    if kernel_backend == KernelBackend.triton:
        num_blocks_per_stride = ceil_divide(stride, BLOCK_SIZE)
        num_strides = divide_if_divisible(x.numel(), x.size(-1))

        with torch.device(x.device):
            swiglu_unchunked_backward_triton_kernel[(num_strides * num_blocks_per_stride,)](
                x_ptr=x,
                output_grad_ptr=output_grad,
                x_grad_ptr=x_grad,
                num_elements=num_elements,
                BLOCK_SIZE=BLOCK_SIZE,
            )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return gate_grad, up_grad
