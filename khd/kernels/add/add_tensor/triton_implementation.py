import triton
import triton.language as tl


@triton.jit
def add_tensor_forward_triton_kernel(x_ptr, y_ptr, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    indices = block_start + tl.arange(0, BLOCK_SIZE)
    mask = indices < num_elements

    x = tl.load(x_ptr + indices, mask=mask)
    y = tl.load(y_ptr + indices, mask=mask)

    output = x + y

    tl.store(output_ptr + indices, output, mask=mask)
