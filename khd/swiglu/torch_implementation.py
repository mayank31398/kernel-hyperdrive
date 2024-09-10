import torch
import torch.nn.functional as F


def swiglu_torch(gate: torch.Tensor, up: torch.Tensor, memory_efficient: bool = False) -> torch.Tensor:
    """swiglu

    Args:
        x (torch.Tensor): input tensor
        y (torch.Tensor): input tensor
        memory_efficient (bool, optional): whether to do an in-place op, will modify `gate` if set to True. Defaults to False.

    Returns:
        torch.Tensor: output tensor
    """

    if memory_efficient:
        F.silu(gate, inplace=True)
        gate *= up

        output = gate
    else:
        output = up * F.silu(gate)

    return output
