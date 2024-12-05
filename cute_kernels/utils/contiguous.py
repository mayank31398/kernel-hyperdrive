from typing import Callable

import torch


def is_hip() -> bool:
    return torch.version.hip is not None


def ensure_contiguous(func: Callable) -> Callable:
    def _contiguous(x):
        return x.contiguous() if isinstance(x, torch.Tensor) else x

    def inner(*args, **kwargs):
        args = [_contiguous(arg) for arg in args]
        kwargs = {k: _contiguous(v) for k, v in kwargs.items()}
        return func(*args, **kwargs)

    return inner
