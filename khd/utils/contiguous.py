import torch
import torch.distributed


def make_contiguous(*args) -> list[torch.Tensor]:
    output = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            arg = arg.contiguous()

        output.append(arg)

    return output


def ensure_same_strides(*args, force_contiguous: bool = False) -> list[torch.Tensor]:
    if force_contiguous:
        output = make_contiguous(*args)
    else:
        mismatch = False
        expected_stride = None

        for arg in args:
            if isinstance(arg, torch.Tensor):
                if expected_stride is None:
                    expected_stride = arg.stride()
                elif arg.stride() != expected_stride:
                    mismatch = True
                    break

        output = make_contiguous(*args) if mismatch else args

    return output
