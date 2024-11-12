import torch
import triton

from ...constants import BLOCK_SIZES_POWERS_OF_2
from ...enums import KernelBackend
from ...utils import CutoTuneParameter, cutotune, ensure_same_strides, get_cartesian_product_cutotune_configs
from .torch_implementation import rmsnorm_torch
from .triton_implementation import rmsnorm_backward_triton_kernel, rmsnorm_forward_triton_kernel


class _RMSNorm_KHD(torch.autograd.Function):
    @staticmethod
    # @cutotune(
    #     configs=get_cartesian_product_cutotune_configs(
    #         kernel_backend=[KernelBackend.triton],
    #         BLOCK_SIZE_B=BLOCK_SIZES_POWERS_OF_2,
    #         BLOCK_SIZE_H=BLOCK_SIZES_POWERS_OF_2,
    #     ),
    #     triggers={"x.dtype"},
    # )
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        kernel_backend: KernelBackend | CutoTuneParameter,
        memory_efficient: bool,
        BLOCK_SIZE_B: int | CutoTuneParameter,
        BLOCK_SIZE_H: int | CutoTuneParameter,
    ) -> torch.Tensor:
        if x.stride(-1) != 1:
            x = x.contiguous()

        assert x.dim() > 1, "x should have more than 1 dimensions"

        if weight is not None:
            assert weight.dim() == 1, "weight should be 1D"
            assert weight.size(-1) == x.size(-1), "hidden size for x and weight tensor is different"
            assert weight.type() == x.type(), "tensors weight and y should have same dtype"

            weight = weight.contiguous()

        hidden_size = x.size(-1)
        num_elements = x.numel() // hidden_size

        x_view = x.view(-1, hidden_size)

        output = torch.empty_like(x)
        rmsnorm_denominator = (
            None if memory_efficient else torch.empty(num_elements, 1, device=x.get_device(), dtype=torch.float32)
        )

        if kernel_backend == KernelBackend.triton:
            grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE_B"]),)

            with torch.device(x.get_device()):
                rmsnorm_forward_triton_kernel[grid](
                    x_ptr=x,
                    x_stride_b=x_view.stride(0),
                    x_stride_h=x_view.stride(1),
                    has_weight=weight is not None,
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

        ctx.memory_efficient = memory_efficient
        ctx.has_weight = weight is not None

        if memory_efficient:
            ctx.save_for_backward(x)
        else:
            ctx.save_for_backward(x, rmsnorm_denominator)

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        memory_efficient = ctx.memory_efficient
        kernel_backend = ctx.kernel_backend_backward
        has_weight = ctx.has_weight

        if memory_efficient:
            (x,) = ctx.saved_tensors
        else:
            x, rmsnorm_denominator = ctx.saved_tensors

        x, output_grad = ensure_same_strides(x, output_grad)

        hidden_size = x.size(-1)
        num_elements = x.numel() // hidden_size

        x_view = x.view(-1, hidden_size)
        output_grad_view = output_grad.view(-1, hidden_size)

        x_grad = torch.empty_like(x)
        weight_grad = torch.empty(hidden_size, x.dtype, device=x.device)

        if kernel_backend == KernelBackend.triton:
            grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE_B"]),)

            with torch.device(x.device):
                rmsnorm_backward_triton_kernel[grid](
                    x_ptr=x,
                    x_stride_b=x_view.stride(0),
                    x_stride_h=x_view.stride(1),
                    has_weight=has_weight,
                    output_grad_ptr=output_grad_view,
                    output_grad_stride_b=output_grad_view.stride(0),
                    output_grad_stride_h=output_grad_view.stride(1),
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

        return x_grad, weight_grad, None, None, None, None, None


def rmsnorm_khd(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    kernel_backend: KernelBackend | CutoTuneParameter = CutoTuneParameter(),
    memory_efficient: bool = False,
    BLOCK_SIZE_B: int | CutoTuneParameter = CutoTuneParameter(),
    BLOCK_SIZE_H: int | CutoTuneParameter = CutoTuneParameter(),
) -> torch.Tensor:
    return _RMSNorm_KHD.apply(x, weight, eps, kernel_backend, memory_efficient, BLOCK_SIZE_B, BLOCK_SIZE_H)
