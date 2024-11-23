import torch
import torch.nn.functional as F


def swiglu_torch(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return up * F.silu(gate)
