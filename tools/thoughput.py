from time import perf_counter

import torch
from tabulate import tabulate

from khd import vector_addition_cuda, vector_addition_torch, vector_addition_triton


n = 100

headers = ["dtype", "torch", "cuda", "triton"]
kernels = [vector_addition_torch, vector_addition_cuda, vector_addition_triton]

table = []

for dtype in [torch.float16, torch.bfloat16, torch.float32]:
    row = [str(dtype)]
    for kernel in kernels:
        # kernel = torch.compile(kernel)
        x = torch.randn(10485760, device=torch.cuda.current_device(), dtype=dtype)
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
