import triton
import triton.language as tl


@triton.jit
def embedding_forward_triton_kernel(
    x_ptr,
    x_stride_b,
    x_stride_s,
    wte_ptr,
    wte_stride_v,
    wte_stride_h,
    logits_ptr,
    logits_stride_b,
    logits_stride_s,
    logits_stride_h,
    B,
    S,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_s = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    block_start_b = pid_b * BLOCK_SIZE_B
    block_start_s = pid_s * BLOCK_SIZE_S
    block_start_h = pid_h * BLOCK_SIZE_H

    indices_b = block_start_b + tl.arange(0, BLOCK_SIZE_B)
    indices_s = block_start_s + tl.arange(0, BLOCK_SIZE_S)
    indices_h = block_start_h + tl.arange(0, BLOCK_SIZE_H)

    mask_b = indices_b < B
    mask_s = indices_s < S
    mask_h = indices_h < H
    mask_bs = mask_b[:, None] & mask_s[None, :]

    x_ptrs = x_ptr + indices_b[:, None] * x_stride_b + indices_s[None, :] * x_stride_s
    x = tl.load(x_ptrs, mask=mask_bs)

    wte_ptrs = wte_ptr + x[:, :, None] * wte_stride_v + indices_h[None, None, :] * wte_stride_h
    word_embeddings = tl.load(wte_ptrs, mask=mask_h[None, None, :])

    logit_ptr = (
        logits_ptr
        + indices_b[:, None, None] * logits_stride_b
        + indices_s[None, :, None] * logits_stride_s
        + indices_h[None, None, :] * logits_stride_h
    )

    tl.store(logit_ptr, word_embeddings, mask=mask_bs[:, :, None] & mask_h[None, None, :])
