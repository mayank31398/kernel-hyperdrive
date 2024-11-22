import torch

from ...utils import ceil_divide


def contiguous_count_naive_kernel(
    x: torch.Tensor, output: torch.Tensor, num_programs: int, C: int, BLOCK_SIZE_B: int, BLOCK_SIZE_C: int
) -> None:
    B = x.numel()

    num_elements_per_program = ceil_divide(B, num_programs)
    num_loops = ceil_divide(num_elements_per_program, BLOCK_SIZE_B)

    max_c = min(C, BLOCK_SIZE_C)
    indices_c = torch.arange(0, max_c, device=x.device)

    for pid in range(num_programs):
        counts = torch.zeros(max_c, dtype=torch.int32, device=x.device)

        for i in range(num_loops):
            start = pid * num_elements_per_program + i * BLOCK_SIZE_B
            end = min(start + BLOCK_SIZE_B, B)

            counts += (x[start:end].unsqueeze(1) == indices_c.unsqueeze(0)).sum(dim=0)

        output[pid, :max_c] = counts
