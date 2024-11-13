import triton
import triton.language as tl


@triton.jit
def add_scalar_forward_triton_kernel(x_ptr, y, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_blocks = tl.num_programs(axis=0)
    is_last_block = pid == num_blocks - 1

    indices = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    if is_last_block:
        mask = indices < num_elements

        x = tl.load(x_ptr + indices, mask=mask)
        tl.store(output_ptr + indices, x + y, mask=mask)
    else:
        x = tl.load(x_ptr + indices)
        tl.store(output_ptr + indices, x + y)
