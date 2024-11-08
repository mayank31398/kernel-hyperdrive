import triton
import triton.language as tl


@triton.jit
def rmsnorm_forward_triton_kernel(
    x_ptr,
    x_stride_b,
    x_stride_h,
    weight_ptr,
    output_ptr,
    num_elements,
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

    denominator = 0
    for pid_h in range(0, triton.cdiv(H, BLOCK_SIZE_H)):
        block_start_h = pid_h * BLOCK_SIZE_H
        indices_h = block_start_h + tl.arange(0, BLOCK_SIZE_H)
        mask_h = indices_h < H

        x_ptrs = x_ptr + indices_b[:, None] * x_stride_b + indices_h[None, :] * x_stride_h
        x = tl.load(x_ptrs, mask=mask_b[:, None] & mask_h[None, :]).to(tl.float32)

        denominator += x * x

    denominator /= H
    denominator += eps
    denominator = tl.rsqrt(denominator)

    for pid_h in range(0, triton.cdiv(H, BLOCK_SIZE_H)):
        block_start_h = pid_h * BLOCK_SIZE_H
        indices_h = block_start_h + tl.arange(0, BLOCK_SIZE_H)
        mask_h = indices_h < H

        x_ptrs = x_ptr + indices_b[:, None] * x_stride_b + indices_h[None, :] * x_stride_h
        x = tl.load(x_ptrs, mask=mask_b[:, None] & mask_h[None, :])

        x *= denominator
        tl.store(x_ptrs, x, mask=mask_b[:, None] & mask_h[None, :])
