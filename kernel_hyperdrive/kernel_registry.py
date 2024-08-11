import importlib
from typing import Callable

from torch.utils.cpp_extension import load as load_cpp_extension


_KERNEL_REGISTRY = {}
_COMPILED_KERNEL_REGISTRY = {}

_MODULE_NAME = "khd"


def register_kernel(name: str, sources: list[str]) -> None:
    global _KERNEL_REGISTRY

    assert name not in _KERNEL_REGISTRY
    _KERNEL_REGISTRY[name] = sources


def get_kernel(name: str) -> Callable:
    global _KERNEL_REGISTRY, _COMPILED_KERNEL_REGISTRY, _MODULE_NAME

    if name not in _COMPILED_KERNEL_REGISTRY:
        assert name in _KERNEL_REGISTRY, f"there is no CUDA kernel identifiable with the name ({name})"

        load_cpp_extension(
            _MODULE_NAME,
            sources=_KERNEL_REGISTRY[name],
            with_cuda=True,
            extra_cflags=["-O3", "-Wall", "-shared", "-fPIC", "-fdiagnostics-color"],
            build_directory="build",
            verbose=True,
        )

        khd = importlib.import_module(_MODULE_NAME)

        _COMPILED_KERNEL_REGISTRY[name] = getattr(khd, name)

    function = _COMPILED_KERNEL_REGISTRY[name]

    return function
