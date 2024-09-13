from typing import Callable

import torch

from ..constants import LIBRARY_NAME


def library_record_function(func: Callable, name: str) -> Callable:
    @torch.profiler.record_function(f"{LIBRARY_NAME}:{name}")
    def f(*args, **kwargs):
        return func(*args, **kwargs)

    return f
