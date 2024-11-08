import triton
import triton.language as tl


@triton.jit
def add_scalar_forward_triton_kernel(x_ptr, y, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    block_indices = block_start + tl.arange(0, BLOCK_SIZE)

    mask = block_indices < num_elements

    x = tl.load(x_ptr + block_indices, mask=mask)

    output = x + y

    tl.store(output_ptr + block_indices, output, mask=mask)
