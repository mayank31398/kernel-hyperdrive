import triton
import triton.language as tl


def _swiglu_forward(gate_ptr, up_ptr, output_ptr, indices, mask):
    gate = tl.load(gate_ptr + indices, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + indices, mask=mask)

    output = up * gate * tl.sigmoid(gate)

    tl.store(output_ptr + indices, output, mask=mask)


@triton.jit
def swiglu_forward_triton_kernel(gate_ptr, up_ptr, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_blocks = tl.num_programs(axis=0)
    is_last_block = pid == num_blocks - 1

    indices = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    if is_last_block:
        mask = indices < num_elements
        _swiglu_forward(gate_ptr=gate_ptr, up_ptr=up_ptr, output_ptr=output_ptr, indices=indices, mask=mask)
    else:
        _swiglu_forward(gate_ptr=gate_ptr, up_ptr=up_ptr, output_ptr=output_ptr, indices=indices, mask=None)
