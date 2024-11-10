from functools import partial
from time import perf_counter

import torch
from tabulate import tabulate

from khd import KernelBackend, add_tensor_khd, add_tensor_torch


n = 100

headers = ["dtype", "torch", "cuda", "triton"]
kernels = [
    add_tensor_torch,
    partial(add_tensor_khd, kernel_backend=KernelBackend.cuda, vector_instruction_width=4, BLOCK_SIZE=1024),
    partial(add_tensor_khd, kernel_backend=KernelBackend.triton, vector_instruction_width=None, BLOCK_SIZE=1024),
]

table = []

for dtype in [torch.float16, torch.bfloat16, torch.float32]:
    row = [str(dtype)]
    for kernel in kernels:
        # kernel = torch.compile(kernel)
        x = torch.randn(10485760, device=torch.cuda.current_device(), dtype=dtype)
        # y = 0.42
        y = torch.randn(10485760, device=torch.cuda.current_device(), dtype=dtype)

        for i in range(n):
            z = kernel(x, y)

        torch.cuda.synchronize()
        s = perf_counter()
        for i in range(n):
            z = kernel(x, y)
        torch.cuda.synchronize()
        e = perf_counter()

        row.append((e - s) / n)
    table.append(row)


print(tabulate(table, headers=headers))
