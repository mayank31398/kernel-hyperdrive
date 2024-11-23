import torch
import triton.language as tl


LIBRARY_NAME = "cute"

CUDA_BLOCK_SIZES_POWERS_OF_2 = [64, 128, 256, 512, 1024]
TRITON_BLOCK_SIZES_POWERS_OF_2 = CUDA_BLOCK_SIZES_POWERS_OF_2 + [2048, 4096, 8192, 16384, 32768, 65536]

MAX_TRITON_BLOCK_SIZE = 65536

TORCH_TO_TRITON_DTYPE = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}
