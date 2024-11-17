import torch

from ...constants import BLOCK_SIZES_POWERS_OF_2
from ...enums import KernelBackend
from ...utils import CutoTuneConfig, CutoTuneParameter, cutotune, ensure_same_strides
from .triton_implementation import rmsnorm_backward_triton_kernel


def _get_cutotune_configs() -> list[CutoTuneConfig]:
    configs = []
    for BLOCK_SIZE_H in BLOCK_SIZES_POWERS_OF_2:
        for BLOCK_SIZE_B in [1, 2, 4, 8, 16, 32] + BLOCK_SIZES_POWERS_OF_2:
            # we only use realistic configs where the block has between 64 and 64k elements
            if 64 < BLOCK_SIZE_B * BLOCK_SIZE_H <= 65536:
                configs.append(
                    CutoTuneConfig(
                        config={
                            "kernel_backend": KernelBackend.triton,
                            "BLOCK_SIZE_B": BLOCK_SIZE_B,
                            "BLOCK_SIZE_H": BLOCK_SIZE_H,
                        },
                        condition=lambda **kwargs: kwargs["x"].size(-1) <= kwargs["BLOCK_SIZE_H"],
                    )
                )

    return configs


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
