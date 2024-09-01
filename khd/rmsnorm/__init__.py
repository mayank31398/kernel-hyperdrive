import torch.nn as nn

from .triton_implementation import RMSNorm_Triton


RMSNorm_Torch = nn.RMSNorm
