import triton
import triton.language as tl


@triton.jit
def swiglu_forward_triton_kernel(gate_ptr, up_ptr, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    indices = block_start + tl.arange(0, BLOCK_SIZE)

    mask = indices < num_elements

    gate = tl.load(gate_ptr + indices, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + indices, mask=mask)

    output = up * gate * tl.sigmoid(gate)

    tl.store(output_ptr + indices, output, mask=mask)


@triton.jit
def swiglu_backward_triton_kernel(
    gate_ptr, up_ptr, output_grad_ptr, gate_grad_ptr, up_grad_ptr, num_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    indices = block_start + tl.arange(0, BLOCK_SIZE)

    mask = indices < num_elements

    gate = tl.load(gate_ptr + indices, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + indices, mask=mask)
    output_grad = tl.load(output_grad_ptr + indices, mask=mask)

    gate_sigmoid = tl.sigmoid(gate)
    gate_silu = gate * gate_sigmoid

    up_grad = output_grad * gate_silu
    gate_grad = output_grad * up * (gate_sigmoid + gate_silu * (1 - gate_sigmoid))

    tl.store(gate_grad_ptr + indices, gate_grad, mask=mask)
    tl.store(up_grad_ptr + indices, up_grad, mask=mask)
