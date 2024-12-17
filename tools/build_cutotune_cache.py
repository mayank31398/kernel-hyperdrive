from typing import Callable

import torch
from tqdm import tqdm

from cute_kernels import (
    add_scalar_cute,
    add_tensor_cute,
    ceil_divide,
    contiguous_count_cute,
    embedding_cute,
    get_1d_tensor_sizes,
    get_2d_tensor_sizes,
    get_dtypes,
    rmsnorm_cute,
    save_cutotune_cache,
    swiglu_cute,
    swiglu_unchunked_cute,
)


def get_tensor_metadata() -> list[tuple[torch.dtype, tuple[int]]]:
    metadata_list = []
    for dtype in get_dtypes():
        for size in get_1d_tensor_sizes():
            metadata_list.append(((size,), dtype))
        for size in get_2d_tensor_sizes():
            metadata_list.append((size, dtype))

    return metadata_list


def forward_backward(kernel: Callable, *args, **kwargs) -> None:
    output = kernel(*args, **kwargs)
    output.sum().backward()


for metadata in tqdm(get_tensor_metadata()):
    size, dtype = metadata
    x = torch.randn(size, dtype=dtype, device=torch.cuda.current_device(), requires_grad=True)

    forward_backward(add_scalar_cute, x, 3)
    forward_backward(add_tensor_cute, x, x)
    forward_backward(swiglu_cute, x, x)

    forward_backward(rmsnorm_cute, x, weight=None, eps=1e-5)
    forward_backward(
        rmsnorm_cute,
        x,
        weight=torch.randn(x.size(-1), dtype=dtype, device=torch.cuda.current_device(), requires_grad=True),
        eps=1e-5,
    )

    forward_backward(
        swiglu_unchunked_cute,
        torch.randn(
            (size[0], ceil_divide(size[1], 2) * 2),
            dtype=metadata[1],
            device=torch.cuda.current_device(),
            requires_grad=True,
        ),
    )

for size in tqdm(get_1d_tensor_sizes()):
    x = torch.randint(0, 64, (size,), device=torch.cuda.current_device(), dtype=torch.long)
    contiguous_count_cute(x, 64)

for dtype in get_dtypes():
    for input_ids_size in [(51, 17), (19, 239), (7, 7537), (9, 1749)]:
        for weight_size in [(7153, 937), (27153, 1937), (97153, 2937), (17153, 31937)]:
            forward_backward(
                embedding_cute,
                input_ids=torch.randint(
                    0, weight_size[0] - 1, input_ids_size, device=torch.cuda.current_device(), dtype=torch.long
                ),
                weight=torch.randn(weight_size, device=torch.cuda.current_device(), dtype=dtype, requires_grad=True),
            )


save_cutotune_cache()
