import triton
import triton.language as tl


@triton.jit
def swiglu_unchunked_forward_triton_kernel(x_ptr, output_ptr, stride, num_blocks_per_stride, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    stride_id = pid // num_blocks_per_stride

    stride_start = stride_id * 2 * stride
    stride_end = stride_start + stride

    block_start = stride_start + (pid % num_blocks_per_stride) * BLOCK_SIZE
    block_indices = block_start + tl.arange(0, BLOCK_SIZE)

    mask = block_indices < stride_end

    up_ptrs = x_ptr + block_indices
    up = tl.load(up_ptrs, mask=mask)
    gate = tl.load(up_ptrs + stride, mask=mask).to(tl.float32)

    output = up * gate * tl.sigmoid(gate)

    output_ptrs = output_ptr + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(output_ptrs, output, mask=mask)