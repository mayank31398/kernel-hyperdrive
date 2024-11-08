import torch
import torch.nn as nn
import triton

from .kernels import rmsnorm_forward_triton_kernel


class _RMSNorm_Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor | None, eps: float | None) -> torch.Tensor:
        assert x.is_cuda, "tensor x is not on GPU"
        assert weight.is_cuda, "tensor weight is not on GPU"

        assert x.is_contiguous(), "tensor x is not a contiguous"
        assert weight.is_contiguous(), "tensor y is not a contiguous"

        normalized_shape = x.size(-1)
        x = x.view(-1, normalized_shape)

        assert weight.dim() == 1, "tensor y should be 1 dimensional"

        assert weight.size(0) == normalized_shape, "both tensors should have same size in the last dimension"
        assert x.type() == weight.type(), "both tensors should have same dtype"

        output = torch.empty_like(x)

        num_elements = x.size(0)
        grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

        rmsnorm_forward_triton_kernel[grid](x, weight, output, num_elements, BLOCK_SIZE=1024)

        return output


def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor | None, eps: float | None) -> torch.Tensor:
    return _RMSNorm_Triton.apply(x, weight, eps)


class RMSNorm_Triton(nn.RMSNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm_triton(x, self.weight, self.eps)
