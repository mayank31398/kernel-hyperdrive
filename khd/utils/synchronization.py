import torch


def device_synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
