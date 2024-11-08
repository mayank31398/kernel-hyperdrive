import torch
import torch.nn as nn
import triton

from .kernels import rmsnorm_forward_triton_kernel


class _RMSNorm_Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor | None, eps: float | None) -> torch.Tensor:
        assert x.is_cuda, "tensor x is not on GPU"
        assert weight.is_cuda, "tensor weight is not on GPU"

        assert x.is_contiguous(), "tensor x is not a contiguous"
        assert weight.is_contiguous(), "tensor y is not a contiguous"

        normalized_shape = x.size(-1)
        x = x.view(-1, normalized_shape)

        assert weight.dim() == 1, "tensor y should be 1 dimensional"

        assert weight.size(0) == normalized_shape, "both tensors should have same size in the last dimension"
        assert x.type() == weight.type(), "both tensors should have same dtype"

        output = torch.empty_like(x)

        num_elements = x.size(0)
        grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

        rmsnorm_forward_triton_kernel[grid](x, weight, output, num_elements, BLOCK_SIZE=1024)

        return output


def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor | None, eps: float | None) -> torch.Tensor:
    return _RMSNorm_Triton.apply(x, weight, eps)


class RMSNorm_Triton(nn.RMSNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm_triton(x, self.weight, self.eps)


import triton
import triton.language as tl


@triton.jit
def rmsnorm_forward_triton_kernel(
    x_ptr,
    x_stride_b,
    x_stride_h,
    output_ptr,
    output_stride_b,
    output_stride_h,
    eps,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)

    block_start_b = pid_b * BLOCK_SIZE_B
    indices_b = block_start_b + tl.arange(0, BLOCK_SIZE_B)
    mask_b = indices_b < B

    denominator = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=tl.float32)
    for pid_h in range(0, triton.cdiv(H, BLOCK_SIZE_H)):
        block_start_h = pid_h * BLOCK_SIZE_H
        indices_h = block_start_h + tl.arange(0, BLOCK_SIZE_H)
        mask_h = indices_h < H
        mask_bh = mask_b[:, None] & mask_h[None, :]

        x_ptrs = x_ptr + indices_b[:, None] * x_stride_b + indices_h[None, :] * x_stride_h
        x = tl.load(x_ptrs, mask=mask_bh).to(tl.float32)

        denominator += x * x

    denominator /= H
    denominator += eps
    denominator = tl.rsqrt(denominator)

    for pid_h in range(0, triton.cdiv(H, BLOCK_SIZE_H)):
        block_start_h = pid_h * BLOCK_SIZE_H
        indices_h = block_start_h + tl.arange(0, BLOCK_SIZE_H)
        mask_h = indices_h < H
        mask_bh = mask_b[:, None] & mask_h[None, :]

        x_ptrs = x_ptr + indices_b[:, None] * x_stride_b + indices_h[None, :] * x_stride_h
        x = tl.load(x_ptrs, mask=mask_bh)

        x *= denominator

        output_ptrs = output_ptr + indices_b[:, None] * output_stride_b + indices_h[None, :] * output_stride_h
        tl.store(output_ptrs, x, mask=mask_bh)
