import os

from torch.utils.cpp_extension import load as load_cpp_extension


def compile_helpers() -> None:
    load_cpp_extension(
        "vector_addition_cuda",
        sources=[
            os.path.join(os.path.dirname(__file__), "vector_addition/cuda_kernel/vector_addition.cpp"),
            os.path.join(os.path.dirname(__file__), "vector_addition/cuda_kernel/vector_addition.cu"),
        ],
        with_cuda=True,
        extra_cflags=["-O3", "-Wall", "-shared", "-fPIC", "-fdiagnostics-color"],
        verbose=True,
    )
