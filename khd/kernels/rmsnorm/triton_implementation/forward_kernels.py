import triton
import triton.language as tl


@triton.jit
def rmsnorm_forward_triton_kernel(
    x_ptr,
    x_stride_b,
    x_stride_h,
    has_weight: tl.constexpr,
    weight_ptr,
    output_ptr,
    output_stride_b,
    output_stride_h,
    eps,
    memory_efficient: tl.constexpr,
    rmsnorm_denominator_ptr,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)

    indices_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    mask_b = indices_b < B

    num_iterations_h = tl.cdiv(H, BLOCK_SIZE_H)

    # when num_iterations_h is 1, we can optimize further
    if num_iterations_h == 1:
        indices_h = tl.arange(0, BLOCK_SIZE_H)
        mask_h = indices_h < H
        mask_bh = mask_b[:, None] & mask_h[None, :]

        x_ptrs = x_ptr + indices_b[:, None] * x_stride_b + indices_h[None, :] * x_stride_h
        x = tl.load(x_ptrs, mask=mask_bh).to(tl.float32)

        denominator = tl.sum(x * x, axis=1, keep_dims=True)
        denominator = tl.rsqrt((denominator / H) + eps)

        if not memory_efficient:
            tl.store(rmsnorm_denominator_ptr + indices_b[:, None], denominator, mask=mask_b[:, None])

        x *= denominator

        if has_weight:
            weight = tl.load(weight_ptr + indices_h, mask=mask_h)
            weight = weight[None, :]
            x *= weight

        output_ptrs = output_ptr + indices_b[:, None] * output_stride_b + indices_h[None, :] * output_stride_h
        tl.store(output_ptrs, x, mask=mask_bh)
    else:
        denominator = tl.zeros((BLOCK_SIZE_B, 1), dtype=tl.float32)

        for pid_h in range(num_iterations_h):
            indices_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
            mask_h = indices_h < H
            mask_bh = mask_b[:, None] & mask_h[None, :]

            x_ptrs = x_ptr + indices_b[:, None] * x_stride_b + indices_h[None, :] * x_stride_h
            x = tl.load(x_ptrs, mask=mask_bh).to(tl.float32)

            denominator += tl.sum(x * x, axis=1, keep_dims=True)

        denominator = tl.rsqrt((denominator / H) + eps)

        if not memory_efficient:
            tl.store(rmsnorm_denominator_ptr + indices_b[:, None], denominator, mask=mask_b[:, None])

        for pid_h in range(num_iterations_h):
            indices_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
            mask_h = indices_h < H
            mask_bh = mask_b[:, None] & mask_h[None, :]

            x_ptrs = x_ptr + indices_b[:, None] * x_stride_b + indices_h[None, :] * x_stride_h
            x = tl.load(x_ptrs, mask=mask_bh).to(tl.float32)

            x *= denominator

            if has_weight:
                weight = tl.load(weight_ptr + indices_h, mask=mask_h)
                weight = weight[None, :]
                x *= weight

            output_ptrs = output_ptr + indices_b[:, None] * output_stride_b + indices_h[None, :] * output_stride_h
            tl.store(output_ptrs, x, mask=mask_bh)
