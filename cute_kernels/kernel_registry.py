import inspect
import os
from typing import Callable

import torch
import yaml
from torch.utils.cpp_extension import load as load_cpp_extension


class _CUDA_JIT:
    module_name = "cute_cuda_kernels"
    build_directory = "build"
    cuda_kernel_registry = {}
    kernel_registry_yaml = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "kernel_registry.yml"), "r"))

    def __init__(self, kernel_name: str) -> None:
        self.kernel_name = kernel_name
        self._kernel = None

    @staticmethod
    def get_kernel(name: str) -> Callable:
        kernel = _CUDA_JIT.cuda_kernel_registry.get(name, None)

        # if kernel is compiled, we return the torch op since its compatible with torch compile
        if kernel is None:
            _CUDA_JIT.compile_kernel(name)
            kernel = _CUDA_JIT.get_kernel(name)

        return kernel

    @torch._dynamo.disable
    @staticmethod
    def compile_kernel(name: str) -> None:
        function_map = []
        all_functions = []
        source_map = []
        for module in _CUDA_JIT.kernel_registry_yaml:
            function_map.append(module["functions"])
            all_functions.extend(module["functions"])

            sources = module["sources"]
            sources = [os.path.join(os.path.dirname(__file__), source) for source in sources]
            source_map.append(sources)

        assert len(all_functions) == len(set(all_functions)), "function names are not unique"

        build_directory = _CUDA_JIT.build_directory
        os.makedirs(build_directory, exist_ok=True)

        # find which files the function belongs to
        for index, functions in enumerate(function_map):
            if name in functions:
                break

        module = load_cpp_extension(
            f"{_CUDA_JIT.module_name}_{index}",
            sources=source_map[index],
            with_cuda=True,
            extra_cflags=["-O3", "-Wall", "-shared", "-fPIC", "-fdiagnostics-color"],
            build_directory=build_directory,
            verbose=True,
        )

        # populate all functions from the file
        for function in function_map[index]:
            _CUDA_JIT.cuda_kernel_registry[function] = getattr(module, function)

    def __call__(self, *args, **kwargs):
        if self._kernel is None:
            self._kernel = _CUDA_JIT.compile_kernel(self.kernel_name)

        return self._kernel(*args, **kwargs)


KernelRegistry = _CUDA_JIT


def cuda_jit(kernel_name: str) -> Callable:
    def inner(function: Callable) -> Callable:
        cuda_function = _CUDA_JIT(kernel_name).__call__

        cuda_function.__signature__ = inspect.signature(function)
        cuda_function.__name__ = function.__name__

    return inner
