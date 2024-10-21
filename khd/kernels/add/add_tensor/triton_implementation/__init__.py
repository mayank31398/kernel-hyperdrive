import torch
import triton

from .....utils import CutoTune, get_default_triton_autotune_configs
from .kernels import add_tensor_forward_triton_kernel


@CutoTune(configs=get_default_triton_autotune_configs(), triggers={"x.dtype"}, overrideables={"BLOCK_SIZE"})
def add_tensor_forward_triton(x: torch.Tensor, y: torch.Tensor, output: torch.Tensor, BLOCK_SIZE: int) -> None:
    num_elements = x.numel()
    grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

    with torch.device(x.device):
        add_tensor_forward_triton_kernel[grid](
            x_ptr=x, y_ptr=y, output_ptr=output, num_elements=num_elements, BLOCK_SIZE=BLOCK_SIZE
        )
