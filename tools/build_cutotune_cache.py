from typing import Callable

import torch
from transformers import set_seed

from cute_kernels import (
    add_scalar_cute,
    add_tensor_cute,
    embedding_cute,
    get_powers_of_2,
    rmsnorm_cute,
    save_cutotune_cache,
    swiglu_cute,
    swiglu_unchunked_cute,
)


def forward_backward(kernel: Callable, *args, **kwargs) -> None:
    output = kernel(*args, **kwargs)
    output.sum().backward()


set_seed(42)
all_dtypes = [torch.float32, torch.float16, torch.bfloat16]


for dtype in all_dtypes:
    for size in [104857600]:
        x = torch.randn(size, dtype=dtype, device=torch.cuda.current_device(), requires_grad=True)

        forward_backward(add_scalar_cute, x, 3)
        forward_backward(add_tensor_cute, x, x)
        forward_backward(swiglu_cute, x, x)

    for size in [(81920, 8192)]:
        forward_backward(
            swiglu_unchunked_cute,
            torch.randn(size, dtype=dtype, device=torch.cuda.current_device(), requires_grad=True),
        )

    for power_of_2 in get_powers_of_2(1, 65536):
        size = (2048, power_of_2)
        print(dtype, size)
        x = torch.randn(size, dtype=dtype, device=torch.cuda.current_device(), requires_grad=True)

        forward_backward(rmsnorm_cute, x, weight=None, eps=1e-5)
        forward_backward(
            rmsnorm_cute,
            x,
            weight=torch.randn(x.size(-1), dtype=dtype, device=torch.cuda.current_device(), requires_grad=True),
            eps=1e-5,
        )

    for input_ids_size in [(32, 4096)]:
        for weight_size in [(131072, 4096)]:
            forward_backward(
                embedding_cute,
                input_ids=torch.randint(
                    0, weight_size[0] - 1, input_ids_size, device=torch.cuda.current_device(), dtype=torch.long
                ),
                weight=torch.randn(weight_size, device=torch.cuda.current_device(), dtype=dtype, requires_grad=True),
            )


save_cutotune_cache()
