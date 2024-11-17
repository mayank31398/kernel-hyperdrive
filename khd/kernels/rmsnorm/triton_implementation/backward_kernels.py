import triton
import triton.language as tl


@triton.jit
def rmsnorm_backward_triton_kernel(
    x_ptr,
    x_stride_b,
    x_stride_h,
    x_dtype: tl.constexpr,
    has_weight: tl.constexpr,
    weight_ptr,
    output_grad_ptr,
    output_grad_stride_b,
    output_grad_stride_h,
    x_grad_ptr,
    weight_grad_ptr,
    eps,
    memory_efficient: tl.constexpr,
    rmsnorm_denominator_ptr,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    tl.device_assert(BLOCK_SIZE_H >= H, "BLOCK_SIZE_H should be more than H")

    num_iterations_b = tl.cdiv(B, BLOCK_SIZE_B)

    if has_weight:
        weight_grad = tl.zeros((BLOCK_SIZE_H,), dtype=tl.float32)

    indices_h = tl.arange(0, BLOCK_SIZE_H)
    mask_h = indices_h < H

    for pid_b in range(num_iterations_b):
        indices_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
        mask_b = indices_b < B

        indices_h = tl.arange(0, BLOCK_SIZE_H)
        mask_h = indices_h < H
        mask_bh = mask_b[:, None] & mask_h[None, :]

        x_ptrs = x_ptr + indices_b[:, None] * x_stride_b + indices_h[None, :] * x_stride_h
        x = tl.load(x_ptrs, mask=mask_bh).to(tl.float32)

        squared_sum = tl.sum(x * x, axis=1)
        inverse_rms = tl.rsqrt(squared_sum / H + eps)

        y_without_weight = x * inverse_rms[:, None]

        output_grad_ptrs = (
            output_grad_ptr + indices_b[:, None] * output_grad_stride_b + indices_h[None, :] * output_grad_stride_h
        )
        output_grad = tl.load(output_grad_ptrs, mask=mask_bh).to(tl.float32)

        if has_weight:
            weight_grad += tl.sum(output_grad * y_without_weight, axis=0)

    if has_weight:
        weight_grad_ptrs = weight_grad_ptr + indices_h
        tl.store(weight_grad_ptrs, weight_grad, mask=mask_h)
