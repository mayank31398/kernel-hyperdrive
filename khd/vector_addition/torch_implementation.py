import torch


def vector_addition_torch(x: torch.Tensor, y: torch.Tensor, memory_efficient: bool = False) -> torch.Tensor:
    """vector addition

    Args:
        x (torch.Tensor): input tensor
        y (torch.Tensor): input tensor
        memory_efficient (bool, optional): whether to do an in-place op, will modify `x` if set to True. Defaults to False.

    Returns:
        torch.Tensor: output tensor
    """

    if memory_efficient:
        x += y
        output = x
    else:
        output = x + y

    return output
