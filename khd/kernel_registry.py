import os
from typing import Callable

import torch
import yaml
from torch.utils.cpp_extension import load as load_cpp_extension


class KernelRegistry:
    module_name = "khd_cuda_kernels"
    build_directory = "build"
    cuda_kernel_registry = {}
    kernel_registry_yaml = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "kernel_registry.yml"), "r"))

    @staticmethod
    def get_kernel(name: str) -> Callable:
        kernel = KernelRegistry.cuda_kernel_registry.get(name, None)

        if kernel is None:
            KernelRegistry.compile_kernel(name)
            kernel = KernelRegistry.get_kernel(name)

        return kernel

    @torch._dynamo.disable
    @staticmethod
    def compile_kernel(name: str) -> Callable:
        function_map = []
        all_functions = []
        source_map = []
        for module in KernelRegistry.kernel_registry_yaml:
            function_map.append(module["functions"])
            all_functions.extend(module["functions"])

            sources = module["sources"]
            sources = [os.path.join(os.path.dirname(__file__), source) for source in sources]
            source_map.append(sources)

        assert len(all_functions) == len(set(all_functions)), "function names are not unique"

        build_directory = KernelRegistry.build_directory
        os.makedirs(build_directory, exist_ok=True)

        # find which files the function belongs to
        for index, functions in enumerate(function_map):
            if name in functions:
                break

        module = load_cpp_extension(
            f"{KernelRegistry.module_name}_{index}",
            sources=source_map[index],
            with_cuda=True,
            extra_cflags=["-O3", "-Wall", "-shared", "-fPIC", "-fdiagnostics-color"],
            build_directory=build_directory,
            verbose=True,
        )

        # populate all functions from the file
        for function in function_map[index]:
            KernelRegistry.cuda_kernel_registry[function] = getattr(module, function)
