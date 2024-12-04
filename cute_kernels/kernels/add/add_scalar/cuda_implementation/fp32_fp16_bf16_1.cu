#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../../../../include/threads.h"

template <typename scalar_t>
__global__ void _add_scalar_forward_cuda_kernel_fp32_fp16_bf16_1(const scalar_t *x,
                                                                 const fp32 y,
                                                                 scalar_t *output,
                                                                 const int64_t num_elements) {
    const uint64 thread_id = get_global_thread_id();

    if (thread_id < num_elements) {
        output[thread_id] = x[thread_id] + y;
    }
}
