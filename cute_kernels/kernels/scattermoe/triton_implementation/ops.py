import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....utils import cute_op
from .kernels import group_triton_kernel, groupXtY_triton_kernel, scatter2scatter_triton_kernel


BLOCK_M = 128
torch._dynamo.config.capture_scalar_outputs = True


# custom op is needed because of https://github.com/pytorch/pytorch/issues/136394
@cute_op(f"{LIBRARY_NAME}::scatter2scatter", mutates_args={"out"})
def scatter2scatter(
    X: torch.Tensor,
    W: torch.Tensor,
    sorted_expert_idxs: torch.Tensor,
    sorted_scattered_idxs: torch.Tensor,
    out: torch.Tensor,
    FAN_OUT: int,
    x_grouped: bool = False,
    y_grouped: bool = False,
) -> None:
    assert sorted_scattered_idxs.size(0) == sorted_expert_idxs.size(0)
    assert sorted_scattered_idxs.size(0) == X.size(0) * FAN_OUT
    assert out.size(0) == sorted_expert_idxs.size(0)
    assert out.size(1) == W.size(-1)

    grid = lambda meta: (
        triton.cdiv(sorted_expert_idxs.size(0), meta["BLOCK_M"]) * triton.cdiv(meta["N"], meta["BLOCK_N"]),
    )

    with torch.device(X.device):
        scatter2scatter_triton_kernel[grid](
            # X_ptr, stride_xm, stride_xk,
            X,
            X.stride(0),
            X.stride(1),
            # W_ptr, stride_we, stride_wk, stride_wn,
            W,
            W.stride(0),
            W.stride(1),
            W.stride(2),
            # Y_ptr, stride_ym, stride_yn,
            out,
            out.stride(0),
            out.stride(1),
            grouped_idx_ptr=sorted_scattered_idxs,
            expert_idxs_ptr=sorted_expert_idxs,
            FAN_OUT=FAN_OUT,
            M=X.size(0),
            K=X.size(1),
            N=out.size(1),
            E=W.size(0),
            BLOCK_M=BLOCK_M,
            ACC_TYPE=tl.float32,
            allow_tf32=torch.backends.cudnn.allow_tf32,
            x_grouped=x_grouped,
            y_grouped=y_grouped,
        )


# custom op is needed because of https://github.com/pytorch/pytorch/issues/136394
@cute_op(f"{LIBRARY_NAME}::group_bwd_W", mutates_args={"DW"})
def group_bwd_W(DY: torch.Tensor, X: torch.Tensor, expert_offsets: torch.Tensor, DW: torch.Tensor, E: int) -> None:
    grid = lambda meta: (E * triton.cdiv(meta["K"], meta["BLOCK_K"]), triton.cdiv(meta["N"], meta["BLOCK_N"]))

    with torch.device(X.device):
        groupXtY_triton_kernel[grid](
            # DY_ptr, stride_dym, stride_dyk,
            DY,
            DY.stride(0),
            DY.stride(1),
            # X_ptr, stride_xm, stride_xn,
            X,
            X.stride(0),
            X.stride(1),
            # DW_ptr, stride_dwe, stride_dwk, stride_dwn,
            DW,
            DW.stride(0),
            DW.stride(1),
            DW.stride(2),
            # expert_offsets_ptr,
            expert_offsets,
            # K: tl.constexpr, N: tl.constexpr,
            N=DY.size(-1),
            K=X.size(-1),
            # ACC_TYPE: tl.constexpr,
            ACC_TYPE=tl.float32,
            allow_tf32=torch.backends.cudnn.allow_tf32,
        )


# custom op is needed because of https://github.com/pytorch/pytorch/issues/136394
@cute_op(f"{LIBRARY_NAME}::group", mutates_args={"out"})
def group(
    A: torch.Tensor,
    sorted_expert_idxs: torch.Tensor,
    out: torch.Tensor,
    coeff: torch.Tensor | None = None,
    fan_out: int = 1,
) -> None:
    N = sorted_expert_idxs.size(0)
    K = A.size(1)
    assert A.size(0) * fan_out == N

    grid = lambda meta: (triton.cdiv(meta["N"], meta["BLOCK_N"]),)

    with torch.device(A.device):
        group_triton_kernel[grid](
            # A_ptr, stride_an, stride_ai,
            A,
            A.stride(0),
            A.stride(1),
            coeff is not None,
            coeff,
            fan_out,
            # Y_ptr, stride_yn, stride_yk,
            out,
            out.stride(0),
            out.stride(1),
            # grouped_idx_ptr,
            sorted_expert_idxs,
            # N: tl.constexpr, K: tl.constexpr,
            N,
            K,
        )
