from time import perf_counter

import torch
from tabulate import tabulate

# from khd import add_scalar_cuda, add_scalar_torch, add_scalar_triton
from khd import add_tensor_cuda, add_tensor_torch, add_tensor_triton


n = 100

headers = ["dtype", "torch", "cuda", "triton"]
# kernels = [add_scalar_torch, add_scalar_cuda, add_scalar_triton]
kernels = [add_tensor_torch, add_tensor_cuda, add_tensor_triton]

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
