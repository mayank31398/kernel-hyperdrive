import yaml
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDAExtension


VERSION = "0.0.0.dev"

kernel_registry: dict = yaml.load(open("khd/kernel_registry.yml", "r"))
all_cuda_sources = list(kernel_registry.values())

setup(
    name="kernels",
    version=VERSION,
    author="Mayank Mishra",
    author_email="mayank31398@ibm.com",
    url="https://github.com/mayank31398/kernels",
    packages=find_packages("./"),
    ext_modules=[CUDAExtension("khd_cuda_kernels", all_cuda_sources)],
    include_package_data=True,
    package_data={"": ["**/*.cu", "**/*.cpp", "**/*.cuh", "**/*.h", "**/*.pyx"]},
)
