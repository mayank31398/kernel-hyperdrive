import importlib
from typing import Callable

from torch.utils.cpp_extension import load as load_cpp_extension


_KERNEL_REGISTRY = {}
_COMPILED_KERNEL_REGISTRY = {}

_MODULE_NAME = "khd_cuda_kernels"


def register_kernel(name: str, sources: list[str]) -> None:
    global _KERNEL_REGISTRY

    if name not in _KERNEL_REGISTRY:
        assert name not in _COMPILED_KERNEL_REGISTRY

        _KERNEL_REGISTRY[name] = sources

        load_cpp_extension(
            _MODULE_NAME,
            sources=sources,
            with_cuda=True,
            extra_cflags=["-O3", "-Wall", "-shared", "-fPIC", "-fdiagnostics-color"],
            build_directory="build",
            verbose=True,
        )

        cuda_kernel_module = importlib.import_module(_MODULE_NAME)
        _COMPILED_KERNEL_REGISTRY[name] = getattr(cuda_kernel_module, name)


def get_kernel(name: str) -> Callable:
    global _COMPILED_KERNEL_REGISTRY
    return _COMPILED_KERNEL_REGISTRY[name]
