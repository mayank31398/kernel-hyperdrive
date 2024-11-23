import torch

from ...utils import ceil_divide


def contiguous_count_naive_kernel(
    x: torch.Tensor, output: torch.Tensor, num_programs: int, B: int, C: int, BLOCK_SIZE_B: int, BLOCK_SIZE_C: int
) -> None:
    num_elements_per_program = ceil_divide(B, num_programs)
    indices_c = torch.arange(0, BLOCK_SIZE_C, device=x.device)

    for pid in range(num_programs):
        program_start = pid * num_elements_per_program
        program_end = min(program_start + num_elements_per_program, B)
        num_elements_in_current_program = program_end - program_start

        num_loops = ceil_divide(num_elements_in_current_program, BLOCK_SIZE_B)
        counts = torch.zeros(BLOCK_SIZE_C, dtype=torch.int32, device=x.device)

        for i in range(num_loops):
            start = program_start + i * BLOCK_SIZE_B
            end = min(start + BLOCK_SIZE_B, program_end)

            x_ = x[start:end].unsqueeze(1).repeat(1, BLOCK_SIZE_C)
            indices_c_ = indices_c.unsqueeze(0).repeat(end - start, 1)

            counts += (x_ == indices_c_).sum(dim=0)

        output[pid] = counts[:C]
