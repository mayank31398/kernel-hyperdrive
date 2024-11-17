import torch

from ...enums import KernelBackend
from ...utils import CutoTuneParameter, ensure_same_strides
from .forward import _forward
from .torch_implementation import rmsnorm_torch
from .triton_implementation import rmsnorm_backward_triton_kernel


class _RMSNorm_KHD(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        memory_efficient: bool,
        kernel_backend_forward: KernelBackend | CutoTuneParameter,
        kernel_backend_backward: KernelBackend | CutoTuneParameter,
        BLOCK_SIZE_B_forward: int | CutoTuneParameter,
        BLOCK_SIZE_B_backward: int | CutoTuneParameter,
        BLOCK_SIZE_H_forward: int | CutoTuneParameter,
        BLOCK_SIZE_H_backward: int | CutoTuneParameter,
    ) -> torch.Tensor:
        output, rmsnorm_denominator = _forward(
            x=x,
            weight=weight,
            eps=eps,
            memory_efficient=memory_efficient,
            kernel_backend=kernel_backend_forward,
            BLOCK_SIZE_B=BLOCK_SIZE_B_forward,
            BLOCK_SIZE_H=BLOCK_SIZE_H_forward,
        )

        has_weight = weight is not None

        ctx.memory_efficient = memory_efficient
        ctx.kernel_backend_backward = kernel_backend_backward
        ctx.has_weight = has_weight
        ctx.eps = eps
        ctx.BLOCK_SIZE_B_backward = BLOCK_SIZE_B_backward
        ctx.BLOCK_SIZE_H_backward = BLOCK_SIZE_H_backward

        tensors_to_save = [x]

        if has_weight:
            tensors_to_save.append(weight)

        if memory_efficient:
            assert rmsnorm_denominator is None
        else:
            tensors_to_save.append(rmsnorm_denominator)

        ctx.save_for_backward(*tensors_to_save)

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        memory_efficient = ctx.memory_efficient
        kernel_backend_backward = ctx.kernel_backend_backward
        has_weight = ctx.has_weight
        eps = ctx.eps
        BLOCK_SIZE_B_backward = ctx.BLOCK_SIZE_B_backward
        BLOCK_SIZE_H_backward = ctx.BLOCK_SIZE_H_backward

        saved_tensors = ctx.saved_tensors

        x = saved_tensors[0]
        weight = saved_tensors[1] if has_weight else None
        rmsnorm_denominator = None if memory_efficient else saved_tensors[2]

        x, output_grad = ensure_same_strides(x, output_grad)

        hidden_size = x.size(-1)
        num_elements = x.numel() // hidden_size

        x_grad = torch.empty_like(x)
        weight_grad = torch.empty(hidden_size, device=x.device, dtype=x.dtype)

        x_view = x.view(-1, hidden_size)
        output_grad_view = output_grad.view(-1, hidden_size)

        if kernel_backend_backward == KernelBackend.triton:
            if BLOCK_SIZE_H_backward < hidden_size:
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
                    BLOCK_SIZE_B=BLOCK_SIZE_B_backward,
                    BLOCK_SIZE_H=BLOCK_SIZE_H_backward,
                )
        else:
            raise ValueError(f"unexpected kernel_backend_backward ({kernel_backend_backward})")

        return x_grad, weight_grad, *[None] * 8


def rmsnorm_khd(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    memory_efficient: bool = False,
    kernel_backend_forward: KernelBackend | CutoTuneParameter = CutoTuneParameter(),
    kernel_backend_backward: KernelBackend | CutoTuneParameter = CutoTuneParameter(),
    BLOCK_SIZE_B_forward: int | CutoTuneParameter = CutoTuneParameter(),
    BLOCK_SIZE_B_backward: int | CutoTuneParameter = CutoTuneParameter(),
    BLOCK_SIZE_H_forward: int | CutoTuneParameter = CutoTuneParameter(),
    BLOCK_SIZE_H_backward: int | CutoTuneParameter = CutoTuneParameter(),
) -> torch.Tensor:
    return _RMSNorm_KHD.apply(
        x,
        weight,
        eps,
        memory_efficient,
        kernel_backend_forward,
        kernel_backend_backward,
        BLOCK_SIZE_B_forward,
        BLOCK_SIZE_B_backward,
        BLOCK_SIZE_H_forward,
        BLOCK_SIZE_H_backward,
    )
