import triton
import triton.language as tl


@triton.jit
def embedding_backward_triton_kernel(
    x_ptr,
    weight_ptr,
    weight_stride_v,
    weight_stride_h,
    output_ptr,
    output_stride_b,
    output_stride_h,
    weight_grad_ptr,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    indices_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    indices_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

    mask_b = indices_b < B
    mask_h = indices_h < H

    x_ptrs = x_ptr + indices_b
    x = tl.load(x_ptrs, mask=mask_b)

    weight_ptrs = weight_ptr + x[:, None] * weight_stride_v + indices_h[None, :] * weight_stride_h
    word_embeddings = tl.load(weight_ptrs, mask=mask_h[None, :])

    output_ptrs = output_ptr + indices_b[:, None] * output_stride_b + indices_h[None, :] * output_stride_h
    tl.store(output_ptrs, word_embeddings, mask=mask_b[:, None] & mask_h[None, :])
