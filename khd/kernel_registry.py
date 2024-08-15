import importlib
from typing import Callable


_COMPILED_KERNEL_REGISTRY = {}
_MODULE_NAME = "khd_cuda_kernels"


def get_kernel(name: str) -> Callable:
    global _COMPILED_KERNEL_REGISTRY, _MODULE_NAME

    if name not in _COMPILED_KERNEL_REGISTRY:
        cuda_kernel_module = importlib.import_module(_MODULE_NAME)
        _COMPILED_KERNEL_REGISTRY[name] = getattr(cuda_kernel_module, name)

    return _COMPILED_KERNEL_REGISTRY[name]
