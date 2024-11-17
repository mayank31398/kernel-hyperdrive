import torch

from ...constants import TORCH_TO_TRITON_DTYPE
from ...enums import KernelBackend
from ...utils import CutoTuneParameter, cutotune, ensure_same_strides
from .configs import _get_cutotune_configs
from .triton_implementation import rmsnorm_backward_triton_kernel


@cutotune(configs=_get_cutotune_configs(), triggers={"x.dtype", "x.size(-1)"})
def _backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    rmsnorm_denominator: torch.Tensor,
    output_grad: torch.Tensor,
    memory_efficient: bool,
    kernel_backend: KernelBackend | CutoTuneParameter,
    BLOCK_SIZE_B: int | CutoTuneParameter,
    BLOCK_SIZE_H: int | CutoTuneParameter,
) -> tuple[torch.Tensor | None]:
    # x already has stride(-1) = 1 from the forward function
    # so we just ensure that x & output_grad have the same strides
    x, output_grad = ensure_same_strides(x, output_grad)

    has_weight = weight is not None

    hidden_size = x.size(-1)
    num_elements = x.numel() // hidden_size

    x_grad = torch.empty_like(x)
    weight_grad = torch.empty(hidden_size, device=x.device, dtype=x.dtype) if has_weight else None

    x_view = x.view(-1, hidden_size)
    output_grad_view = output_grad.view(-1, hidden_size)

    if kernel_backend == KernelBackend.triton:
        if BLOCK_SIZE_H < hidden_size:
            raise ValueError(f"hidden_size should be more than the BLOCK_SIZE_H_backward")

        grid = (1,)

        with torch.device(x.device):
            rmsnorm_backward_triton_kernel[grid](
                x_ptr=x,
                x_stride_b=x_view.stride(0),
                x_stride_h=x_view.stride(1),
                x_dtype=TORCH_TO_TRITON_DTYPE[x.dtype],
                has_weight=has_weight,
                weight_ptr=weight,
                output_grad_ptr=output_grad,
                output_grad_stride_b=output_grad_view.stride(0),
                output_grad_stride_h=output_grad_view.stride(1),
                x_grad_ptr=x_grad,
                weight_grad_ptr=weight_grad,
                eps=eps,
                memory_efficient=memory_efficient,
                rmsnorm_denominator_ptr=rmsnorm_denominator,
                B=num_elements,
                H=hidden_size,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
            )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return x_grad, weight_grad
