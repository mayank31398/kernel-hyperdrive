import torch
import torch.nn.functional as F


def embedding_torch(input_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.embedding(input_ids, weight)
