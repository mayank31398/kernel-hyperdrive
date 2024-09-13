from typing import Callable, Iterable, Union

import torch


def library_custom_op(
    name: str, fn: Callable | None = None, /, *, mutates_args: Union[str, Iterable[str]]
) -> Callable:
    @torch.library.custom_op(name, mutates_args=mutates_args)
    def f(*args, **kwargs):
        return fn(*args, **kwargs)

    return f
