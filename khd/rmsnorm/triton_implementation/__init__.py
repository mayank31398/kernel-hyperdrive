import torch.nn as nn
from torch import Tensor

from .ops import rmsnorm_triton


class RMSNorm_Triton(nn.RMSNorm):
    def forward(self, x: Tensor) -> Tensor:
        return rmsnorm_triton(x, self.weight, self.eps)
