import torch


def vector_addition_torch(x: torch.Tensor, y: torch.Tensor, in_place: bool = False) -> torch.Tensor:
    """vector addition

    Args:
        x (torch.Tensor): input tensor
        y (torch.Tensor): input tensor
        in_place (bool, optional): whether to do an in-place op, will modify `x` if set to True. Defaults to False.

    Returns:
        torch.Tensor: output tensor
    """

    if in_place:
        x += y
        output = x
    else:
        output = x + y

    return output
