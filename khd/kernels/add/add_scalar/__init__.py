import torch
import triton

from ....enums import KernelBackend
from ....utils import CutoTune, get_default_cuda_cutotune_configs, get_default_triton_cutotune_configs
from .cuda_implementation import add_scalar_forward_cuda_kernel, add_scalar_forward_cuda_kernel_compileable
from .torch_implementation import add_scalar_torch
from .triton_implementation import add_scalar_forward_triton_kernel


class _AddScalar_KHD(torch.autograd.Function):
    @staticmethod
    @CutoTune(
        configs=get_default_cuda_cutotune_configs(
            vectorized_loop_size_8_condition=lambda **kwargs: kwargs["x"].dtype in [torch.float16, torch.bfloat16]
        )
        + get_default_triton_cutotune_configs(vectorized_loop_size=None)
    )
    def forward(
        ctx, x: torch.Tensor, y: float, kernel_backend: KernelBackend, BLOCK_SIZE: int | None = None
    ) -> torch.Tensor:
        if y == 0:
            return x

        assert x.is_cuda, "tensor x is not on GPU"

        output = torch.empty_like(x)

        if kernel_backend == KernelBackend.cuda:
            if torch.compiler.is_compiling():
                add_scalar_forward_cuda_kernel_compileable(x=x, y=y, output=output, BLOCK_SIZE=BLOCK_SIZE)
            else:
                add_scalar_forward_cuda_kernel(x=x, y=y, output=output, BLOCK_SIZE=BLOCK_SIZE)
        elif kernel_backend == KernelBackend.triton:
            num_elements = x.numel()
            grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

            with torch.device(x.device):
                add_scalar_forward_triton_kernel[grid](
                    x_ptr=x, y=y, output_ptr=output, num_elements=num_elements, BLOCK_SIZE=BLOCK_SIZE
                )
        else:
            raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        return output_grad, None, None, None


def add_scalar_khd(
    x: torch.Tensor, y: float, kernel_backend: KernelBackend, BLOCK_SIZE: int | None = None
) -> torch.Tensor:
    return _AddScalar_KHD.apply(x, y, kernel_backend, BLOCK_SIZE)
