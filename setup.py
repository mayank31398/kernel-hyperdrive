from setuptools import find_packages, setup


VERSION = "0.0.0.dev"

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
