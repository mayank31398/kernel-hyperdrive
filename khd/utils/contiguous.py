import torch
import torch.distributed


def make_contiguous(*args) -> list[torch.Tensor]:
    output = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            arg = arg.contiguous()

        output.append(arg)

    return output


def ensure_same_strides(*args, expected_stride: tuple[int], force_contiguous: bool = False) -> list[torch.Tensor]:
    if force_contiguous:
        output = make_contiguous(*args)
    else:
        mismatch = False
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.stride() != expected_stride:
                mismatch = True
                break

        output = make_contiguous(*args) if mismatch else args

    return output
