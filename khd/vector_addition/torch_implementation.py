import torch


def vector_addition_torch(x: torch.Tensor, y: torch.Tensor, in_place: bool) -> torch.Tensor:
    if in_place:
        x += y
        output = x
    else:
        output = x + y

    return output
