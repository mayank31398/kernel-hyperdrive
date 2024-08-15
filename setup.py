import yaml
from setuptools import find_packages, setup


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
    include_package_data=True,
    package_data={"": ["**/*.cu", "**/*.cpp", "**/*.cuh", "**/*.h", "**/*.pyx"]},
)
