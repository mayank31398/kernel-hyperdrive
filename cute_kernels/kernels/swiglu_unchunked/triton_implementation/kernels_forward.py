import triton
import triton.language as tl


@triton.jit
def swiglu_unchunked_forward_triton_kernel(
    x_ptr, output_ptr, B, H, BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_H: tl.constexpr
):
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    indices_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    indices_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

    stride = H / 2
    indices_bh = indices_b[:, None] * H + indices_h[None, :]
    mask_bh = (indices_b < B)[:, None] & (indices_h < H)[None, :]

    up_ptrs = x_ptr + indices_bh
    up = tl.load(up_ptrs, mask=mask_bh)
    gate = tl.load(up_ptrs + stride, mask=mask_bh).to(tl.float32)

    output = up * gate * tl.sigmoid(gate)
    tl.store(output_ptr + indices_bh, output, mask=mask_bh)
