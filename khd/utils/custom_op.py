from typing import Callable, Iterable, Union

import torch


def library_custom_op(
    func: Callable,
    name: str,
    mutates_args: Union[str, Iterable[str]],
) -> Callable:
    @torch.library.custom_op(name, mutates_args=mutates_args)
    def f(*args, **kwargs):
        return func(*args, **kwargs)

    return f
