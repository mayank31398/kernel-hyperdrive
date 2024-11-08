import torch
import triton

from ....constants import BLOCK_SIZES_POWERS_OF_2
from ....enums import KernelBackend
from ....utils import CutoTuneParameter, cutotune, ensure_same_strides, get_cartesian_product_cutotune_configs
from .cuda_implementation import add_tensor_forward_cuda_kernel, add_tensor_forward_cuda_kernel_compileable
from .torch_implementation import add_tensor_torch
from .triton_implementation import add_tensor_forward_triton_kernel


class _AddTensor_KHD(torch.autograd.Function):
    @staticmethod
    @cutotune(
        configs=(
            get_cartesian_product_cutotune_configs(
                kernel_backend=[KernelBackend.cuda], vectorized_loop_size=[1, 2, 4], BLOCK_SIZE=BLOCK_SIZES_POWERS_OF_2
            )
            if torch.cuda.is_available()
            else []
        )
        + (
            get_cartesian_product_cutotune_configs(
                kernel_backend=[KernelBackend.cuda],
                vectorized_loop_size=[8],
                BLOCK_SIZE=BLOCK_SIZES_POWERS_OF_2,
                condition=lambda **kwargs: kwargs["x"].dtype in [torch.float16, torch.bfloat16],
            )
            if torch.cuda.is_available()
            else []
        )
        + get_cartesian_product_cutotune_configs(
            kernel_backend=[KernelBackend.triton], vectorized_loop_size=[None], BLOCK_SIZE=BLOCK_SIZES_POWERS_OF_2
        ),
        triggers={"x.dtype"},
    )
    def forward(
        ctx,
        x: torch.Tensor,
        y: torch.Tensor,
        kernel_backend: KernelBackend | CutoTuneParameter,
        vectorized_loop_size: int | CutoTuneParameter,
        BLOCK_SIZE: int | CutoTuneParameter,
    ) -> torch.Tensor:
        assert x.size() == y.size(), "tensors x and y should have same shape"
        assert x.type() == y.type(), "tensors x and y should have same dtype"

        x, y = ensure_same_strides(x, y, expected_stride=x.stride())
        output = torch.empty_like(x)

        if kernel_backend == KernelBackend.cuda:
            if torch.compiler.is_compiling():
                add_tensor_forward_cuda_kernel_compileable(
                    x=x, y=y, output=output, vectorized_loop_size=vectorized_loop_size, BLOCK_SIZE=BLOCK_SIZE
                )
            else:
                add_tensor_forward_cuda_kernel(
                    x=x, y=y, output=output, vectorized_loop_size=vectorized_loop_size, BLOCK_SIZE=BLOCK_SIZE
                )
        elif kernel_backend == KernelBackend.triton:
            assert vectorized_loop_size is None

            num_elements = x.numel()
            grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

            with torch.device(x.device):
                add_tensor_forward_triton_kernel[grid](
                    x_ptr=x, y_ptr=y, output_ptr=output, num_elements=num_elements, BLOCK_SIZE=BLOCK_SIZE
                )
        else:
            raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        return output_grad, output_grad, None, None, None


def add_tensor_khd(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel_backend: KernelBackend | CutoTuneParameter = CutoTuneParameter(),
    vectorized_loop_size: int | CutoTuneParameter = CutoTuneParameter(),
    BLOCK_SIZE: int | CutoTuneParameter = CutoTuneParameter(),
) -> torch.Tensor:
    return _AddTensor_KHD.apply(x, y, kernel_backend, vectorized_loop_size, BLOCK_SIZE)
