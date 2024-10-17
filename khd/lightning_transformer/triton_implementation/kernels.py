import triton
import triton.language as tl

from .embedding import _get_word_embeddings


@triton.jit
def lightning_transformer_triton_kernel(
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

    word_embeddings = _get_word_embeddings(
        x_ptr=x_ptr,
        x_stride_b=x_stride_b,
        x_stride_s=x_stride_s,
        wte_ptr=wte_ptr,
        wte_stride_v=wte_stride_v,
        wte_stride_h=wte_stride_h,
        indices_b=indices_b,
        indices_s=indices_s,
        indices_h=indices_h,
        mask_bs=mask_bs,
        mask_h=mask_h,
    )

    logit_ptr = (
        logits_ptr
        + indices_b[:, None, None] * logits_stride_b
        + indices_s[None, :, None] * logits_stride_s
        + indices_h[None, None, :] * logits_stride_h
    )
    tl.store(logit_ptr, word_embeddings, mask=mask_bs[:, :, None] & mask_h[None, None, :])