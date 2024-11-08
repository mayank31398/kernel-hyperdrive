import torch
import triton

from ...constants import BLOCK_SIZES_POWERS_OF_2
from ...enums import KernelBackend
from ...utils import CutoTuneParameter, cutotune, get_cartesian_product_cutotune_configs
from .torch_implementation import RMSNorm_Torch
from .triton_implementation import rmsnorm_forward_triton_kernel


class _RMSNorm_KHD(torch.autograd.Function):
    @staticmethod
    @cutotune(
        configs=get_cartesian_product_cutotune_configs(
            kernel_backend=[KernelBackend.triton], BLOCK_SIZE=BLOCK_SIZES_POWERS_OF_2
        ),
        triggers={"x.dtype"},
    )
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        kernel_backend: KernelBackend | CutoTuneParameter,
        BLOCK_SIZE_B: int | CutoTuneParameter,
        BLOCK_SIZE_H: int | CutoTuneParameter,
    ) -> torch.Tensor:
        assert weight.dim() == 1
        assert weight.size(-1) == x.size(-1), ""

        weight = weight.contiguous()

        if x.stride(-1) != 1:
            x = x.contiguous()

        hidden_size = x.size(-1)
        num_elements = x.numel() / hidden_size

        output = torch.empty_like(x)

        if kernel_backend == KernelBackend.triton:
            grid = lambda meta: (
                triton.cdiv(num_elements, meta["BLOCK_SIZE_B"]),
                triton.cdiv(hidden_size, meta["BLOCK_SIZE_H"]),
            )

            with torch.device(x.device):
                rmsnorm_forward_triton_kernel[grid](
                    x_ptr=x,
                    x_stride_b=x.stride(0),
                    x_stride_h=x.stride(1),
                    output_ptr=output,
                    output_stride_b=output.stride(0),
                    output_stride_h=output.stride(1),
                    eps=eps,
                    B=num_elements,
                    H=hidden_size,
                    BLOCK_SIZE_B=BLOCK_SIZE_B,
                    BLOCK_SIZE_H=BLOCK_SIZE_H,
                )
        else:
            raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

        return output


def rmsnorm_khd(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    kernel_backend: KernelBackend | CutoTuneParameter = CutoTuneParameter(),
    BLOCK_SIZE_B: int | CutoTuneParameter = CutoTuneParameter(),
    BLOCK_SIZE_H: int | CutoTuneParameter = CutoTuneParameter(),
) -> torch.Tensor:
    return _RMSNorm_KHD.apply(x, weight, kernel_backend, BLOCK_SIZE_B, BLOCK_SIZE_H)
