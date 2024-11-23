import torch


def add_scalar_forward_naive_kernel(
    num_programs: int, x: torch.Tensor, y: float, output: torch.Tensor, num_elements: int, BLOCK_SIZE: int
) -> None:
    for pid in range(num_programs):
        start = pid * BLOCK_SIZE
        end = min(start + BLOCK_SIZE, num_elements)

        output[start:end] = x[start:end] + y[start:end]
