import triton
import triton.language as tl


@triton.jit
def embedding_forward_triton_kernel(
    x_ptr,
    wte_ptr,
    wte_stride_v,
    wte_stride_h,
    output_ptr,
    output_stride_b,
    output_stride_h,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    block_start_b = pid_b * BLOCK_SIZE_B
    block_start_h = pid_h * BLOCK_SIZE_H

    indices_b = block_start_b + tl.arange(0, BLOCK_SIZE_B)
    indices_h = block_start_h + tl.arange(0, BLOCK_SIZE_H)

    mask_b = indices_b < B
    mask_h = indices_h < H
    mask_bs = mask_b[:, None] & mask_h[None, :]

    x_ptrs = x_ptr + indices_b
    x = tl.load(x_ptrs, mask=mask_bs)

    wte_ptrs = wte_ptr + x[:, None] * wte_stride_v + indices_h[None, :] * wte_stride_h
    word_embeddings = tl.load(wte_ptrs, mask=mask_h[None, :])

    output_ptrs = output_ptr + indices_b[:, None] * output_stride_b + indices_h[None, :] * output_stride_h
    tl.store(output_ptrs, word_embeddings, mask=mask_bs[:, None] & mask_h[None, :])
