import torch


def add_tensor_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """vector addition

    Args:
        x (torch.Tensor): input tensor
        y (torch.Tensor): input tensor

    Returns:
        torch.Tensor: output tensor
    """

    return x + y
