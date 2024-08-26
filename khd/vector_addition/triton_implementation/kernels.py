import triton
import triton.language as tl


@triton.jit
def vector_addition_forward_triton_kernel(x_ptr, y_ptr, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_blocks = tl.num_programs(axis=0)

    block_start = pid * BLOCK_SIZE
    block_indices = block_start + tl.arange(0, BLOCK_SIZE)

    is_last_block = pid == num_blocks - 1
    mask = block_indices < num_elements

    if is_last_block:
        x = tl.load(x_ptr + block_indices, mask=mask)
        y = tl.load(y_ptr + block_indices, mask=mask)
    else:
        x = tl.load(x_ptr + block_indices)
        y = tl.load(y_ptr + block_indices)

    output = x + y

    if is_last_block:
        tl.store(output_ptr + block_indices, output, mask=mask)
    else:
        tl.store(output_ptr + block_indices, output)
