import torch
import torch.nn as nn


def vector_addition_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


class VectorAddition_Torch(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return vector_addition_torch(x, y)
