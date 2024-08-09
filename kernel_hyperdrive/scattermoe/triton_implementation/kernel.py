import triton
import triton.language as tl


BLOCK_M = 128


@triton.autotune(
    configs=[triton.Config({"BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=4)],
    key=["M", "N", "K"],
)
@triton.heuristics(
    {
        "NO_K_MASK": lambda args: (args["K"] % args["BLOCK_K"]) == 0,
        "NO_N_MASK": lambda args: (args["N"] % args["BLOCK_N"]) == 0,
    }
)
@triton.jit
def scatter2scatter_triton_kernel(
    X_ptr,
    stride_xm,
    stride_xk,
    W_ptr,
    stride_we,
    stride_wk,
    stride_wn,
    Y_ptr,
    stride_ym,
    stride_yn,
    grouped_idx_ptr,
    expert_idxs_ptr,
    block_start_idx_ptr,
    FAN_OUT: tl.constexpr,
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    OUT_M,
    allow_tf32: tl.constexpr,
    x_grouped: tl.constexpr,
    y_grouped: tl.constexpr,
    NO_K_MASK: tl.constexpr,
    NO_N_MASK: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    N_BLOCK_COUNT = tl.cdiv(N, BLOCK_N)
    M_block_id = pid // N_BLOCK_COUNT
    N_block_id = pid % N_BLOCK_COUNT
    M_range = tl.arange(0, BLOCK_M)
    block_start_idx = tl.load(block_start_idx_ptr + M_block_id)
    # M_block = tl.max_contiguous((block_start_idx + M_range) % OUT_M, BLOCK_M)
    M_block = tl.max_contiguous(block_start_idx + M_range, BLOCK_M)
    E_idxs = tl.load(expert_idxs_ptr + M_block, mask=M_block < (FAN_OUT * M), other=E)
    E_idx = tl.min(E_idxs)
    E_mask = E_idxs == E_idx
    M_idx = tl.load(grouped_idx_ptr + M_block, mask=E_mask, other=0)
    if x_grouped:
        M_in_idx = M_block
    else:
        M_in_idx = M_idx // FAN_OUT

    if y_grouped:
        M_out_idx = M_block
    else:
        M_out_idx = M_idx

    K_block = tl.arange(0, BLOCK_K)

    N_block = N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    N_mask = N_block < N
    # N_block = tl.max_contiguous(tl.multiple_of(N_block % N, BLOCK_N), BLOCK_N)
    # N_block = N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)

    X_blk_ptrs = X_ptr + M_in_idx[:, None] * stride_xm + K_block[None, :] * stride_xk
    W_blk_ptrs = W_ptr + K_block[:, None] * stride_wk + N_block[None, :] * stride_wn + E_idx * stride_we

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    iters = tl.cdiv(K, BLOCK_K)
    for K_block_id in range(0, iters):
        if NO_K_MASK:
            x = tl.load(X_blk_ptrs, mask=E_mask[:, None])
            if NO_N_MASK or K_block_id < (iters - 1):
                w = tl.load(W_blk_ptrs)
            else:
                w = tl.load(W_blk_ptrs, mask=N_mask[None, :])
        else:
            K_mask = (K_block_id * BLOCK_K + K_block) < K
            x = tl.load(X_blk_ptrs, mask=E_mask[:, None] & K_mask[None, :])
            w = tl.load(W_blk_ptrs, mask=K_mask[:, None] & N_mask[None, :])
        X_blk_ptrs += BLOCK_K * stride_xk
        W_blk_ptrs += BLOCK_K * stride_wk
        acc += tl.dot(x, w, allow_tf32=allow_tf32, out_dtype=ACC_TYPE)

    Y_blk_ptrs = Y_ptr + (M_out_idx[:, None] * stride_ym + N_block[None, :] * stride_yn)
    tl.store(Y_blk_ptrs, acc, mask=E_mask[:, None] & N_mask[None, :])


@triton.autotune(
    configs=[triton.Config({"BLOCK_N": 128, "BLOCK_K": 128, "BLOCK_M": 32}, num_stages=4, num_warps=4)],
    key=["M", "N", "K"],
)
@triton.heuristics(
    {
        "NO_K_MASK": lambda args: (args["K"] % args["BLOCK_K"]) == 0,
        "NO_N_MASK": lambda args: (args["N"] % args["BLOCK_N"]) == 0,
    }
)
@triton.jit
def groupXtY_triton_kernel(
    DY_ptr,
    stride_dym,
    stride_dyk,
    X_ptr,
    stride_xm,
    stride_xn,
    DW_ptr,
    stride_dwe,
    stride_dwk,
    stride_dwn,
    expert_offsets_ptr,
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    allow_tf32: tl.constexpr,
    NO_K_MASK: tl.constexpr,
    NO_N_MASK: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)
    num0 = tl.num_programs(0)
    num1 = tl.num_programs(1)
    pid1, pid0 = tl.swizzle2d(pid1, pid0, num1, num0, 128)

    K_BLOCK_COUNT = tl.cdiv(K, BLOCK_K)
    E_idx = pid0 // K_BLOCK_COUNT
    K_block_id = pid0 % K_BLOCK_COUNT
    N_block_id = pid1

    if E_idx == 0:
        start_idx = 0
    else:
        start_idx = tl.load(expert_offsets_ptr + E_idx - 1).to(tl.int32)
    end_idx = tl.load(expert_offsets_ptr + E_idx).to(tl.int32)

    if end_idx > start_idx:
        M_block = tl.max_contiguous(start_idx + tl.arange(0, BLOCK_M), BLOCK_M)

        K_block = K_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
        K_mask = K_block < K
        K_block = tl.max_contiguous(tl.multiple_of(K_block % K, BLOCK_K), BLOCK_K)

        N_block = N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        N_mask = N_block < N
        N_block = tl.max_contiguous(tl.multiple_of(N_block % N, BLOCK_N), BLOCK_N)

        M_idxs = M_block
        xt_blk_ptrs = X_ptr + K_block[:, None] * stride_xn + M_idxs[None, :] * stride_xm
        dy_blk_ptrs = DY_ptr + M_idxs[:, None] * stride_dym + N_block[None, :] * stride_dyk

        acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=ACC_TYPE)
        iters = tl.cdiv(end_idx - start_idx, BLOCK_M)
        for i in range(0, iters):
            M_mask = (i * BLOCK_M + M_block) < end_idx
            if NO_K_MASK:
                xt = tl.load(xt_blk_ptrs, mask=M_mask[None, :])
            else:
                xt = tl.load(xt_blk_ptrs, mask=K_mask[:, None] & M_mask[None, :])
            if NO_N_MASK:
                dy = tl.load(dy_blk_ptrs, mask=M_mask[:, None])
            else:
                dy = tl.load(dy_blk_ptrs, mask=M_mask[:, None] & N_mask[None, :])
            # acc += tl.dot(xt, dy, out_dtype=ACC_TYPE, allow_tf32=allow_tf32)
            xt_blk_ptrs += BLOCK_M * stride_xm
            dy_blk_ptrs += BLOCK_M * stride_dym
            acc += tl.dot(xt, dy, out_dtype=ACC_TYPE, allow_tf32=allow_tf32)

        DW_blk_ptrs = DW_ptr + E_idx * stride_dwe + K_block[:, None] * stride_dwk + N_block[None, :] * stride_dwn
        acc = acc.to(DW_blk_ptrs.dtype.element_ty)
        tl.store(DW_blk_ptrs, acc, mask=K_mask[:, None] & N_mask[None, :])


@triton.autotune(configs=[triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_stages=4, num_warps=4)], key=["K"])
@triton.heuristics({"NO_K_MASK": lambda args: (args["K"] % args["BLOCK_K"]) == 0})
@triton.jit
def group_triton_kernel(
    src_ptr,
    stride_sn,
    stride_sk,
    has_coeff: tl.constexpr,
    coeff_ptr,
    FAN_OUT: tl.constexpr,
    tgt_ptr,
    stride_tn,
    stride_ti,
    grouped_idx_ptr,
    N,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NO_K_MASK: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    N_block_id = pid
    N_blk = N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    N_mask = N_blk < N
    N_blk = tl.max_contiguous(tl.multiple_of(N_blk % N, BLOCK_N), BLOCK_N)
    N_idx = tl.load(grouped_idx_ptr + N_blk, mask=N_mask, other=0)

    K_blk = tl.arange(0, BLOCK_K)
    src_blk_ptrs = src_ptr + (N_idx // FAN_OUT)[:, None] * stride_sn + K_blk[None, :] * stride_sk
    tgt_blk_ptrs = tgt_ptr + N_blk[:, None] * stride_tn + K_blk[None, :] * stride_ti

    if has_coeff:
        c = tl.load(coeff_ptr + N_idx, mask=N_mask)[:, None]

    iters = tl.cdiv(K, BLOCK_K)
    for i in range(0, iters):
        if NO_K_MASK or i < iters - 1:
            block = tl.load(src_blk_ptrs, mask=N_mask[:, None])
            if has_coeff:
                block *= c
            tl.store(tgt_blk_ptrs, block, mask=N_mask[:, None])

        else:
            K_mask = (i * BLOCK_K + K_blk) < K
            mask = N_mask[:, None] & K_mask[None, :]
            block = tl.load(src_blk_ptrs, mask=mask)
            if has_coeff:
                block *= c
            tl.store(tgt_blk_ptrs, block, mask=mask)
        src_blk_ptrs += BLOCK_K * stride_sk
        tgt_blk_ptrs += BLOCK_K * stride_ti
