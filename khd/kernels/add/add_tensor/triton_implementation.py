import triton
import triton.language as tl


@triton.jit
def add_tensor_forward_triton_kernel(x_ptr, y_ptr, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_blocks = tl.num_programs(axis=0)
    is_last_block = pid == num_blocks - 1

    block_start = pid * BLOCK_SIZE
    indices = block_start + tl.arange(0, BLOCK_SIZE)

    if is_last_block:
        mask = indices < num_elements

        x = tl.load(x_ptr + indices, mask=mask)
        y = tl.load(y_ptr + indices, mask=mask)
    else:
        x = tl.load(x_ptr + indices)
        y = tl.load(y_ptr + indices)

    output = x + y

    if is_last_block:
        tl.store(output_ptr + indices, output, mask=mask)
    else:
        tl.store(output_ptr + indices, output)
