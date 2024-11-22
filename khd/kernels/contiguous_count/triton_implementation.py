import triton
import triton.language as tl


@triton.jit
def contiguous_count_triton_kernel(
    x_ptr, output_ptr, output_stride_b, B, C, BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_C: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    num_elements_per_program = tl.cdiv(B, num_programs)
    num_loops = tl.cdiv(num_elements_per_program, BLOCK_SIZE_B)

    indices_c = tl.arange(0, BLOCK_SIZE_C)
    mask_c = indices_c < C

    counts = tl.zeros((BLOCK_SIZE_C,), dtype=tl.int32)

    for i in range(num_loops):
        start = pid * num_elements_per_program + i * BLOCK_SIZE_B
        indices_b = start + tl.arange(0, BLOCK_SIZE_B)
        mask_b = indices_b < B

        x = tl.load(x_ptr + indices_b, mask=mask_b)

        equal = x[:, None] == indices_c[None, :]
        counts += tl.sum(equal, axis=0)

    output_ptrs = output_ptr + pid * output_stride_b + indices_c
    tl.store(output_ptrs, counts, mask=mask_c)
