from enum import Enum


class KernelBackend(Enum):
    cuda = "cuda"
    triton = "triton"
    naive = "naive"
