import triton
import triton.language as tl


@triton.jit
def contiguous_count_triton_kernel(
    x_ptr, output_ptr, output_stride_b, num_elements, start, end, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    indices = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = indices < num_elements

    comparator = tl.arange(start, end)

    x_ptrs = x_ptr + indices
    x = x.load(x_ptrs, mask=mask)

    equal = x[:, None] == comparator[None, :]
    count = tl.sum(equal, axis=0, keep_dims=True)

    output_ptrs = output_ptr + pid * output_stride_b + comparator - start
    tl.store(output_ptrs, count)
