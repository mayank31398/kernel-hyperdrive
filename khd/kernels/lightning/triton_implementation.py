import triton
import triton.language as tl


@triton.jit
def lightning_transformer_forward_triton_kernel(
    x_ptr,
    wte_ptr,
    wte_stride_v,
    wte_stride_h,
    output_ptr,
    output_stride_b,
    output_stride_s,
    output_stride_h,
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
    mask_bsh = mask_bs[:, :, None] & mask_h[None, None, :]

    x_ptrs = x_ptr + indices_b[:, None] * S + indices_s[None, :]
    x = tl.load(x_ptrs, mask=mask_bs)

    x = _get_word_embeddings(
        input_ids=x,
        wte_ptr=wte_ptr,
        wte_stride_v=wte_stride_v,
        wte_stride_h=wte_stride_h,
        indices_h=indices_h,
        mask_h=mask_h,
    )

    output_ptrs = (
        output_ptr
        + indices_b[:, None, None] * output_stride_b
        + indices_s[None, :, None] * output_stride_s
        + indices_h[None, None, :] * output_stride_h
    )
    tl.store(output_ptrs, x, mask=mask_bsh)


@triton.jit
def _get_word_embeddings(
    input_ids,
    wte_ptr,
    wte_stride_v,
    wte_stride_h,
    indices_h,
    mask_h,
):
    wte_ptrs = wte_ptr + input_ids[:, :, None] * wte_stride_v + indices_h[None, None, :] * wte_stride_h
    word_embeddings = tl.load(wte_ptrs, mask=mask_h[None, None, :])
    return word_embeddings
