import torch


def device_synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_sm_count(device: torch.device) -> int:
    if device.type == "cuda":
        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    elif device.type == "xpu":
        sm_count = torch.xpu.get_device_properties(device).gpu_subslice_count

    return sm_count
