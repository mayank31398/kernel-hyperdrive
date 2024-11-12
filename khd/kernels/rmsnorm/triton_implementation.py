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
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)

    block_start_b = pid_b * BLOCK_SIZE_B
    indices_b = block_start_b + tl.arange(0, BLOCK_SIZE_B)
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
            block_start_h = pid_h * BLOCK_SIZE_H
            indices_h = block_start_h + tl.arange(0, BLOCK_SIZE_H)
            mask_h = indices_h < H
            mask_bh = mask_b[:, None] & mask_h[None, :]

            x_ptrs = x_ptr + indices_b[:, None] * x_stride_b + indices_h[None, :] * x_stride_h
            x = tl.load(x_ptrs, mask=mask_bh).to(tl.float32)

            denominator += tl.sum(x * x, axis=1, keep_dims=True)

        denominator = tl.rsqrt((denominator / H) + eps)

        for pid_h in range(num_iterations_h):
            block_start_h = pid_h * BLOCK_SIZE_H
            indices_h = block_start_h + tl.arange(0, BLOCK_SIZE_H)
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


@triton.jit
def rmsnorm_backward_triton_kernel(
    x_ptr,
    x_stride_b,
    x_stride_h,
    has_weight: tl.constexpr,
    weight_ptr,
    output_grad_ptr,
    output_grad_stride_b,
    output_grad_stride_h,
    x_grad_ptr,
    weight_grad_ptr,
    eps,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    num_iterations_h = tl.cdiv(H, BLOCK_SIZE_H)
    weight_grad = tl.zeros((H,), dtype=tl.float32)

    for pid_b in range(BLOCK_SIZE_B):
        block_start_b = pid_b * BLOCK_SIZE_B
        indices_b = block_start_b + tl.arange(0, BLOCK_SIZE_B)
        mask_b = indices_b < B

        # when num_iterations_h is 1, we can optimize further
        if num_iterations_h == 1:
            indices_h = tl.arange(0, H)

            x_ptrs = x_ptr + indices_b[:, None] * x_stride_b + indices_h[None, :] * x_stride_h
            x = tl.load(x_ptrs, mask=mask_b[:, None]).to(tl.float32)

            denominator = tl.sum(x * x, axis=1, keep_dims=True)
            denominator = tl.rsqrt((denominator / H) + eps)
            y = x * denominator

            output_grad_ptrs = (
                output_grad_ptr + indices_b[:, None] * output_grad_stride_b + indices_h[None, :] * output_grad_stride_h
            )
            output_grad = tl.load(output_grad_ptrs, mask=mask_b[:, None]).to(tl.float32)

            if has_weight:
                weight_grad += tl.sum(y * output_grad, axis=0)
        else:
            pass

    if num_iterations_h == 1:
        weight_grad_ptrs = weight_grad_ptr + tl.arange(0, H)
        tl.store(weight_grad_ptrs, weight_grad)
    else:
        pass
