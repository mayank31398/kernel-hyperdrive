import torch
import triton

from ...constants import BLOCK_SIZES_POWERS_OF_2
from ...enums import KernelBackend
from ...utils import CutoTuneParameter, cutotune, get_cartesian_product_cutotune_configs
from .cuda_implementation import swiglu_backward_cuda_kernel, swiglu_backward_cuda_kernel_compileable
from .triton_implementation import swiglu_backward_triton_kernel


@cutotune(
    configs=(
        get_cartesian_product_cutotune_configs(
            kernel_backend=[KernelBackend.cuda],
            vector_instruction_width=[1, 2, 4],
            BLOCK_SIZE=BLOCK_SIZES_POWERS_OF_2,
        )
        if torch.cuda.is_available()
        else []
    )
    + (
        get_cartesian_product_cutotune_configs(
            kernel_backend=[KernelBackend.cuda],
            vector_instruction_width=[8],
            BLOCK_SIZE=BLOCK_SIZES_POWERS_OF_2,
            condition=lambda **kwargs: kwargs["gate"].dtype in [torch.float16, torch.bfloat16],
        )
        if torch.cuda.is_available()
        else []
    )
    + get_cartesian_product_cutotune_configs(
        kernel_backend=[KernelBackend.triton],
        vector_instruction_width=[None],
        BLOCK_SIZE=BLOCK_SIZES_POWERS_OF_2,
    ),
    triggers={"gate.dtype"},
)
def _backward(
    gate: torch.Tensor,
    up: torch.Tensor,
    output_grad: torch.Tensor,
    kernel_backend: KernelBackend | CutoTuneParameter,
    vector_instruction_width: int | CutoTuneParameter,
    BLOCK_SIZE: int | CutoTuneParameter,
) -> tuple[torch.Tensor]:
    gate_grad = torch.empty_like(gate)
    up_grad = torch.empty_like(up)

    if kernel_backend == KernelBackend.cuda:
        if torch.compiler.is_compiling():
            swiglu_backward_cuda_kernel_compileable(
                gate=gate,
                up=up,
                output_grad=output_grad,
                gate_grad=gate_grad,
                up_grad=up_grad,
                vector_instruction_width=vector_instruction_width,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        else:
            swiglu_backward_cuda_kernel(
                gate=gate,
                up=up,
                output_grad=output_grad,
                gate_grad=gate_grad,
                up_grad=up_grad,
                vector_instruction_width=vector_instruction_width,
                BLOCK_SIZE=BLOCK_SIZE,
            )
    elif kernel_backend == KernelBackend.triton:
        num_elements = gate.numel()
        grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

        with torch.device(gate.device):
            swiglu_backward_triton_kernel[grid](
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
