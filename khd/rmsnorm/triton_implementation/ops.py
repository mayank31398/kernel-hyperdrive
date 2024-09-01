import torch
import triton

from .kernels import rmsnorm_triton_kernel


class _RMSNorm_Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor | None, eps: float | None) -> torch.Tensor:
        assert x.is_cuda, "tensor x is not on GPU"
        assert w.is_cuda, "tensor w is not on GPU"

        assert x.is_contiguous(), "tensor x is not a contiguous"
        assert w.is_contiguous(), "tensor y is not a contiguous"

        normalized_shape = x.size(-1)
        x = x.view(-1, normalized_shape)

        assert w.dim() == 1, "tensor y should be 1 dimensional"

        assert w.size(0) == normalized_shape, "both tensors should have same size in the last dimension"
        assert x.type() == w.type(), "both tensors should have same dtype"

        output = torch.empty_like(x)

        num_elements = x.numel()
        grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

        rmsnorm_triton_kernel[grid](x, y, output, num_elements, BLOCK_SIZE=1024)

        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return output_grad, output_grad


def rmsnorm_triton(x: torch.Tensor, w: torch.Tensor | None, eps: float | None) -> torch.Tensor:
    return _RMSNorm_Triton.apply(x, w, eps)
