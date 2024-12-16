import random

import torch


def get_all_devices() -> list[torch.device]:
    return [torch.device("cpu"), torch.device("cuda")]


def get_dtypes() -> list[torch.dtype]:
    return [torch.float32, torch.float16, torch.bfloat16]


def get_1d_tensor_sizes() -> list[tuple[int]]:
    sizes = set()
    # powers of 2
    for i in range(15):
        start = 2**i
        for j in range(10):
            sizes.add(start + j)
    # not powers of 2
    for _ in range(50):
        sizes.add(3000 + random.randint(-1000, 1000))
    return sizes


def get_2d_tensor_sizes(static_batch_size: int | None = None) -> list[tuple[int]]:
    sizes = set()
    # powers of 2
    for i in range(15):
        start = 2**i
        for j in range(10):
            sizes.add((start + j if static_batch_size is None else static_batch_size, start + j))
    # not powers of 2
    for _ in range(50):
        sizes.add(
            (
                3000 + random.randint(-1000, 1000) if static_batch_size is None else static_batch_size,
                3000 + random.randint(-1000, 1000),
            )
        )
    return sizes
