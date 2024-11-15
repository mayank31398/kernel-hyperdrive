import torch

from ....enums import KernelBackend
from ....utils import CutoTuneParameter
from .forward import _forward


class _AddScalar_KHD(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        y: float,
        kernel_backend: KernelBackend | CutoTuneParameter,
        vector_instruction_width: int | CutoTuneParameter,
        BLOCK_SIZE: int | CutoTuneParameter,
    ) -> torch.Tensor:
        return _forward(
            ctx=ctx,
            x=x,
            y=y,
            kernel_backend=kernel_backend,
            vector_instruction_width=vector_instruction_width,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        return output_grad, None, None, None, None


def add_scalar_khd(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel_backend: KernelBackend | CutoTuneParameter = CutoTuneParameter(),
    vector_instruction_width: int | CutoTuneParameter = CutoTuneParameter(),
    BLOCK_SIZE: int | CutoTuneParameter = CutoTuneParameter(),
) -> torch.Tensor:
    return _AddScalar_KHD.apply(x, y, kernel_backend, vector_instruction_width, BLOCK_SIZE)
