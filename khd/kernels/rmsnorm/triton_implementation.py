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
    for pid_h in range(0, tl.cdiv(H, BLOCK_SIZE_H)):
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

    for pid_h in range(0, tl.cdiv(H, BLOCK_SIZE_H)):
        block_start_h = pid_h * BLOCK_SIZE_H
        indices_h = block_start_h + tl.arange(0, BLOCK_SIZE_H)
        mask_h = indices_h < H
        mask_bh = mask_b[:, None] & mask_h[None, :]

        x_ptrs = x_ptr + indices_b[:, None] * x_stride_b + indices_h[None, :] * x_stride_h
        x = tl.load(x_ptrs, mask=mask_bh)

        x *= denominator

        output_ptrs = output_ptr + indices_b[:, None] * output_stride_b + indices_h[None, :] * output_stride_h
        tl.store(output_ptrs, x, mask=mask_bh)
