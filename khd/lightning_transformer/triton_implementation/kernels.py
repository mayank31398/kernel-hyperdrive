import triton
import triton.language as tl


@triton.jit
def _get_word_embeddings(
    x_ptr,
    x_stride_b,
    x_stride_s,
    wte_ptr,
    wte_stride_v,
    wte_stride_h,
    indices_b,
    indices_s,
    indices_h,
    mask_bs,
    mask_h,
):
    x_ptrs = x_ptr + indices_b[:, None] * x_stride_b + indices_s[None, :] * x_stride_s
    x = tl.load(x_ptrs, mask=mask_bs)

    wte_ptrs = wte_ptr + x * wte_stride_v + indices_h[None, :] * wte_stride_h
    word_embeddings = tl.load(wte_ptrs, mask=mask_h)

    return word_embeddings


@triton.jit
def lightning_transformer_triton_kernel(
    x_ptr,
    x_stride_b,
    x_stride_s,
    wte_ptr,
    wte_stride_v,
    wte_stride_h,
    logits_ptr,
    B,
    S,
    H,
    BLOCK_SIZE_B,
    BLOCK_SIZE_S,
    BLOCK_SIZE_H,
):
    pid_b = tl.program_id(axis=0)
    pid_s = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    block_start_b = pid_b * BLOCK_SIZE_B
    block_start_s = pid_s * BLOCK_SIZE_S
    block_start_h = pid_h * BLOCK_SIZE_H

    indices_b = block_start_b + tl.arange(0, BLOCK_SIZE_B)
    indices_s = block_start_s + tl.arange(0, BLOCK_SIZE_S)
    indices_h = block_start_h + tl.arange(0, BLOCK_SIZE_H)

    mask_b = indices_b < B
    mask_s = indices_s < S
    mask_h = indices_h < H
    mask_bs = mask_b[:, None] & mask_s[None, :]

    word_embeddings = _get_word_embeddings(
        x_ptr=x_ptr,
        x_stride_b=x_stride_b,
        x_stride_s=x_stride_s,
        wte_ptr=wte_ptr,
        wte_stride_v=wte_stride_v,
        wte_stride_h=wte_stride_h,
        indices_b=indices_b,
        indices_h=indices_h,
        mask_bs=mask_bs,
        mask_h=mask_h,
    )

    tl.store(logits_ptr, word_embeddings, mask=mask_bs[:, :, None] & mask_h[None, None, :])


# 2024-09-17 20:40:42,391 - [INFO    ] ▶ step = 10, train-loss = 10.5773, train-grad_norm = 106.5810, learning_rate = 1.0000e-04, train-FLOPs = 114.9260, train-billion_tokens_per_day = 3.6498, train-step_time (sec) = 6.2055
# 2024-09-17 20:41:06,413 - [INFO    ] ▶ step = 20, train-loss = 7.4504, train-grad_norm = 14.6408, learning_rate = 2.0000e-04, train-FLOPs = 296.8974, train-billion_tokens_per_day = 9.4289, train-step_time (sec) = 2.4021
# 2024-09-17 20:41:30,541 - [INFO    ] ▶ step = 30, train-loss = 6.2820, train-grad_norm = 4.9204, learning_rate = 3.0000e-04, train-FLOPs = 295.5880, train-billion_tokens_per_day = 9.3873, train-step_time (sec) = 2.4127


# 2024-09-17 20:53:02,727 - [INFO    ] ▶ step = 60, train-loss = 5.6262, train-grad_norm = 4.1566, learning_rate = 6.0000e-04, train-FLOPs = 293.9193, train-billion_tokens_per_day = 9.3343, train-step_time (sec) = 2.4264
# 2024-09-17 20:53:26,994 - [INFO    ] ▶ step = 70, train-loss = 5.3426, train-grad_norm = 3.7945, learning_rate = 7.0000e-04, train-FLOPs = 293.8937, train-billion_tokens_per_day = 9.3335, train-step_time (sec) = 2.4267
# 2024-09-17 20:53:51,450 - [INFO    ] ▶ step = 80, train-loss = 5.0332, train-grad_norm = 4.0429, learning_rate = 8.0000e-04, train-FLOPs = 291.6207, train-billion_tokens_per_day = 9.2613, train-step_time (sec) = 2.4456
# 2024-09-17 20:54:15,574 - [INFO    ] ▶ step = 90, train-loss = 4.7266, train-grad_norm = 3.5853, learning_rate = 9.0000e-04, train-FLOPs = 295.6335, train-billion_tokens_per_day = 9.3888, train-step_time (sec) = 2.4124
# 2024-09-17 20:54:39,677 - [INFO    ] ▶ step = 100, train-loss = 4.4518, train-grad_norm = 3.4109, learning_rate = 1.0000e-03, train-FLOPs = 295.8840, train-billion_tokens_per_day = 9.3967, train-step_time (sec) = 2.4103
# 2024-09-17 20:55:03,768 - [INFO    ] ▶ step = 110, train-loss = 4.3172, train-grad_norm = 3.5594, learning_rate = 1.1000e-03, train-FLOPs = 296.0478, train-billion_tokens_per_day = 9.4019, train-step_time (sec) = 2.4090
