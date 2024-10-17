import triton
import triton.language as tl


@triton.jit
def _get_word_embeddings(
    x_ptr,
    x_stride_b,
    x_stride_s,
    wte_ptr,
    wte_stride_v,
    wte_stride_h,
    indices_b,
    indices_s,
    indices_h,
    mask_bs,
    mask_h,
):
    x_ptrs = x_ptr + indices_b[:, None] * x_stride_b + indices_s[None, :] * x_stride_s
    x = tl.load(x_ptrs, mask=mask_bs)

    wte_ptrs = wte_ptr + x[:, :, None] * wte_stride_v + indices_h[None, None, :] * wte_stride_h
    word_embeddings = tl.load(wte_ptrs, mask=mask_h[None, None, :])

    return word_embeddings
