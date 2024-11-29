from typing import Callable, Iterable, Sequence

import torch


def torch_custom_op(
    name: str,
    fn: Callable | None = None,
    /,
    *,
    mutates_args: str | Iterable[str],
    device_types: str | Sequence[str] | None = None,
    schema: str | None = None,
    fake_fn: Callable | None = None,
) -> Callable:
    compileable_fn = torch.library.custom_op(
        name, fn, mutates_args=mutates_args, device_types=device_types, schema=schema
    )

    if fake_fn is not None:
        compileable_fn.register_fake(fake_fn)

    def inner(*args, **kwargs):
        if torch.compiler.is_compiling():
            output = compileable_fn(*args, **kwargs)
        else:
            output = fn(*args, **kwargs)

        return output

    return inner
