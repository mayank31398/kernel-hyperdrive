import triton
import triton.language as tl


@triton.jit
def contiguous_count_triton_kernel(
    x_ptr, output_ptr, output_stride_b, num_elements, start, end, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    num_elements_per_program = tl.cdiv(num_elements, num_programs)
    num_loops = tl.cdiv(num_elements_per_program, BLOCK_SIZE)

    counts = tl.zeros((end - start,), dtype=tl.int32)
    comparator = tl.arange(start, end)

    for i in range(num_loops):
        start = pid * num_elements_per_program + i * BLOCK_SIZE
        end = start + BLOCK_SIZE

        indices = tl.arange(start, end)
        mask = indices < num_elements

        x_ptrs = x_ptr + indices
        x = tl.load(x_ptrs, mask=mask)

        equal = x[:, None] == comparator[None, :]
        counts += tl.sum(equal, axis=0)

    output_ptrs = output_ptr + pid * output_stride_b + comparator - start
    tl.store(output_ptrs, counts)
