import torch

from ...utils import ceil_divide


def contiguous_count_naive_kernel(
    x: torch.Tensor, output: torch.Tensor, sm_count: int, C: int, BLOCK_SIZE_B: int, BLOCK_SIZE_C: int
) -> None:
    B = x.numel()

    num_programs = ceil_divide(B, sm_count)
    num_elements_per_program = ceil_divide(B, num_programs)
    num_loops = ceil_divide(num_elements_per_program, BLOCK_SIZE_B)

    indices_c = torch.arange(0, min(C, BLOCK_SIZE_C))

    for pid in range(num_programs):
        for i in range(num_loops):
            start = pid * num_elements_per_program + i * BLOCK_SIZE_B
            end = min(start + BLOCK_SIZE_B, B)

            output[pid, : min(C, BLOCK_SIZE_C)] += (x[start:end].unsqueeze(1) == indices_c.unsqueeze(0)).sum(dim=0)
