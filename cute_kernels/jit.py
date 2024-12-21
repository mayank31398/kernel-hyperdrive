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


@torch._dynamo.disable
def compile_cpp(name: str) -> None:
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


def cpp_jit(kernel_name: str) -> Callable:
    kernel = None
    args_spec = None

    def _run(*args, **kwargs):
        nonlocal kernel

        if kernel is None:
            kernel = compile_cpp(kernel_name)

        full_args = []
        full_args.extend(args)
        for variable_name in args_spec.args[len(args) :]:
            full_args.append(kwargs[variable_name])

        return kernel(*full_args)

    def inner(function: Callable) -> Callable:
        _run.__signature__ = inspect.signature(function)
        _run.__name__ = function.__name__

        nonlocal args_spec
        args_spec = inspect.getfullargspec(function)

        return _run

    return inner
