import torch
import triton

from ...constants import BLOCK_SIZES_POWERS_OF_2
from ...enums import KernelBackend
from ...utils import CutoTuneConfig, CutoTuneParameter, cutotune
from .triton_implementation import rmsnorm_forward_triton_kernel


def _get_cutotune_configs() -> list[CutoTuneConfig]:
    configs = []
    for BLOCK_SIZE_H in BLOCK_SIZES_POWERS_OF_2:
        for BLOCK_SIZE_B in [1, 2, 4, 8, 16, 32] + BLOCK_SIZES_POWERS_OF_2:
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
def _forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    memory_efficient: bool,
    kernel_backend: KernelBackend | CutoTuneParameter,
    BLOCK_SIZE_B: int | CutoTuneParameter,
    BLOCK_SIZE_H: int | CutoTuneParameter,
) -> tuple[torch.Tensor | None]:
    if x.stride(-1) != 1:
        x = x.contiguous()

    assert x.dim() > 1, "x should have more than 1 dimensions"

    has_weight = weight is not None

    if has_weight:
        assert weight.dim() == 1, "weight should be 1D"
        assert weight.size(-1) == x.size(-1), "hidden size for x and weight tensor is different"
        assert weight.type() == x.type(), "tensors weight and y should have same dtype"

        weight = weight.contiguous()

    hidden_size = x.size(-1)
    num_elements = x.numel() // hidden_size

    x_view = x.view(-1, hidden_size)

    output = torch.empty_like(x)
    rmsnorm_denominator = None if memory_efficient else torch.empty(num_elements, device=x.device, dtype=torch.float32)

    if kernel_backend == KernelBackend.triton:
        if BLOCK_SIZE_H < hidden_size:
            raise ValueError(f"hidden_size should be more than the BLOCK_SIZE_H")

        grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE_B"]),)

        with torch.device(x.device):
            rmsnorm_forward_triton_kernel[grid](
                x_ptr=x,
                x_stride_b=x_view.stride(0),
                x_stride_h=x_view.stride(1),
                has_weight=has_weight,
                weight_ptr=weight,
                output_ptr=output,
                output_stride_b=output.stride(0),
                output_stride_h=output.stride(1),
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

    return output, rmsnorm_denominator
