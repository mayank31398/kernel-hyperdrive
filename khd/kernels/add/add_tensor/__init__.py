import torch

from ....enums import KernelBackend
from ....utils import AutoTune
from .cuda_implementation import add_tensor_cuda
from .torch_implementation import add_tensor_torch
from .triton_implementation import add_tensor_triton


_KERNELS = {KernelBackend.cuda: add_tensor_cuda, KernelBackend.triton: add_tensor_triton}


@AutoTune(configs=[{"kernel_backend": KernelBackend.cuda}, {"kernel_backend": KernelBackend.triton}])
def add_tensor_generic(x: torch.Tensor, y: torch.Tensor, kernel_backend: KernelBackend) -> torch.Tensor:
    """tensor addition

    Args:
        x (torch.Tensor): input tensor
        y (torch.Tensor): input tensor

    Returns:
        torch.Tensor: output tensor
    """

    return _KERNELS[kernel_backend](x, y)
