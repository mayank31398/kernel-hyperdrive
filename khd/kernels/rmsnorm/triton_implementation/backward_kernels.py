import triton
import triton.language as tl


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
    memory_efficient: tl.constexpr,
    rmsnorm_denominator_ptr,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    num_iterations_h = tl.cdiv(H, BLOCK_SIZE_H)

    if has_weight:
        weight_grad = tl.zeros((1, BLOCK_SIZE_H), dtype=tl.float32)

    for pid_b in range(BLOCK_SIZE_B):
        indices_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
        mask_b = indices_b < B

        # when num_iterations_h is 1, we can optimize further
        if num_iterations_h == 1:
            indices_h = tl.arange(0, BLOCK_SIZE_H)
            mask_h = indices_h < H
            mask_bh = mask_b[:, None] & mask_h[None, :]

            x_ptrs = x_ptr + indices_b[:, None] * x_stride_b + indices_h[None, :] * x_stride_h
            x = tl.load(x_ptrs, mask=mask_bh).to(tl.float32)

            if memory_efficient:
                denominator = tl.sum(x * x, axis=1, keep_dims=True)
                denominator = tl.rsqrt((denominator / H) + eps)
            else:
                denominator = tl.load(rmsnorm_denominator_ptr + indices_b, mask=mask_b)

            y_without_weight = x * denominator

            output_grad_ptrs = (
                output_grad_ptr + indices_b[:, None] * output_grad_stride_b + indices_h[None, :] * output_grad_stride_h
            )
            output_grad = tl.load(output_grad_ptrs, mask=mask_bh).to(tl.float32)

            x_grad = output_grad * denominator * (1 - y_without_weight * y_without_weight)

            if has_weight:
                weight_ptrs = weight_ptr + indices_h
                weight = tl.load(weight_ptrs, mask=mask_h)

                x_grad *= weight[None, :]
                weight_grad += tl.sum(y_without_weight * output_grad, axis=0, keep_dims=True)

            x_grad_ptrs = x_grad_ptr + indices_b[:, None] * x_stride_b + indices_h[None, :] * x_stride_h
            tl.store(x_grad_ptrs, x_grad, mask=mask_bh)
        else:
            if memory_efficient:
                denominator = tl.zeros((BLOCK_SIZE_B, 1), dtype=tl.float32)

            for pid_h in range(BLOCK_SIZE_H):
                indices_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
                mask_h = indices_h < H
                mask_bh = mask_b[:, None] & mask_h[None, :]

                x_ptrs = x_ptr + indices_b[:, None] * x_stride_b + indices_h[None, :] * x_stride_h
                x = tl.load(x_ptrs, mask=mask_bh).to(tl.float32)

                if memory_efficient:
                    denominator += tl.sum(x * x, axis=1, keep_dims=True)

            if memory_efficient:
                denominator = tl.rsqrt((denominator / H) + eps)
            else:
                denominator = tl.load(rmsnorm_denominator_ptr + indices_b, mask=mask_b)

            for pid_h in range(BLOCK_SIZE_H):
                indices_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
                mask_h = indices_h < H
                mask_bh = mask_b[:, None] & mask_h[None, :]

                x_ptrs = x_ptr + indices_b[:, None] * x_stride_b + indices_h[None, :] * x_stride_h
                x = tl.load(x_ptrs, mask=mask_bh).to(tl.float32)

                y_without_weight = x * denominator

                output_grad_ptrs = (
                    output_grad_ptr
                    + indices_b[:, None] * output_grad_stride_b
                    + indices_h[None, :] * output_grad_stride_h
                )
                output_grad = tl.load(output_grad_ptrs, mask=mask_bh).to(tl.float32)

                x_grad = output_grad * denominator * (1 - y_without_weight * y_without_weight)

                if has_weight:
                    weight_ptrs = weight_ptr + indices_h
                    weight = tl.load(weight_ptrs, mask=mask_h)

                    x_grad *= weight[None, :]
                    weight_grad += tl.sum(y_without_weight * output_grad, axis=0, keep_dims=True)

                x_grad_ptrs = x_grad_ptr + indices_b[:, None] * x_stride_b + indices_h[None, :] * x_stride_h
                tl.store(x_grad_ptrs, x_grad, mask=mask_bh)

    if has_weight:
        for pid_h in range(num_iterations_h):
            indices_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
            mask_h = indices_h < H

            tl.store(weight_grad_ptr + indices_h[None, :], weight_grad, mask=mask_h[None, :])
