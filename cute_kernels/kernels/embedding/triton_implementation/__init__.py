import torch

from ....constants import LIBRARY_NAME
from ....utils import ceil_divide, cute_op
from .kernels_forward import embedding_forward_triton_kernel


_KERNEL_NAME = "embedding_forward_triton"


@cute_op(f"{LIBRARY_NAME}::{_KERNEL_NAME}", mutates_args={"output"})
def embedding_forward_triton(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> None:
    num_elements = input_ids.numel()
    hidden_size = weight.size(-1)

    with torch.device(input_ids.device):
        embedding_forward_triton_kernel[
            (ceil_divide(num_elements, BLOCK_SIZE_B), ceil_divide(hidden_size, BLOCK_SIZE_H))
        ](
            x_ptr=input_ids,
            weight_ptr=weight,
            output_ptr=output,
            B=num_elements,
            H=hidden_size,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
