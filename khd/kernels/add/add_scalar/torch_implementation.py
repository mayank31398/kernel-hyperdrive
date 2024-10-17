import torch


def add_scalar_torch(x: torch.Tensor, y: float) -> torch.Tensor:
    """vector addition

    Args:
        x (torch.Tensor): input tensor
        y (float): input scalar

    Returns:
        torch.Tensor: output tensor
    """

    return x + y
