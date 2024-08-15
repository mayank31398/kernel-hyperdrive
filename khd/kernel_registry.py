import os
from typing import Callable

import yaml
from torch.utils.cpp_extension import load as load_cpp_extension


_MODULE_NAME = "khd_cuda_kernels"
_CUDA_KERNEL_MODULE = None
_CUDA_KERNEL_SOURCES = []


class KernelRegistry:
    def __init__(self) -> None:
        global _CUDA_KERNEL_MODULE, _CUDA_KERNEL_SOURCES

        registry: dict = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "kernel_registry.yml"), "r"))

        functions = []
        for module in registry:
            functions.extend(module["functions"])

        assert len(functions) == len(set(functions)), "function names are not unique"

        for module in registry:
            _CUDA_KERNEL_SOURCES.extend(module["sources"])

        _CUDA_KERNEL_MODULE = load_cpp_extension(
            _MODULE_NAME,
            sources=_CUDA_KERNEL_SOURCES,
            with_cuda=True,
            extra_cflags=["-O3", "-Wall", "-shared", "-fPIC", "-fdiagnostics-color"],
            build_directory="build",
            verbose=True,
        )

    def get_sources(self) -> list[str]:
        global _CUDA_KERNEL_SOURCES
        return _CUDA_KERNEL_SOURCES

    def get_kernel(self, name: str) -> Callable:
        global _CUDA_KERNEL_MODULE
        return getattr(_CUDA_KERNEL_MODULE, name)
