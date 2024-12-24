#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../../../include/dtypes/all.h"
#include "../../../include/threads.h"

#define MAX_ALLOWED_C 16384

inline __device__ void _init_shared_memory(uint32 *x, const int &C, const uint32 &thread_id) {
    const int num_loops = (C + blockDim.x - 1) / blockDim.x;
    // clang-format off
    #pragma unroll
    // clang-format on
    for (int i = 0; i < num_loops; i++) {
        x[thread_id + i * blockDim.x] = 0;
    }
}

template <typename scalar_t, int vector_instruction_width>
__global__ void _contiguous_count_cuda_kernel(const scalar_t *x,
                                              const scalar_t *output,
                                              const uint64 num_elements,
                                              const uint32 C) {
    const uint64 thread_id = get_global_thread_id();

    __shared__ uint32 output_shared[MAX_ALLOWED_C];
    _init_shared_memory(output_shared, C, thread_id);
    __syncthreads();

    const uint32 *x_vec = (uint32 *)&((uint32_4 *)x)[thread_id];

    // clang-format off
    #pragma unroll
    // clang-format on
    for (int i = 0; i < 4; i++) {
        uint32 *x_local = (uint32 *)x_vec[thread_id];
        x_local[i];
    }
}

void contiguous_count_cuda(const torch::Tensor &x, const torch::Tensor &output, const int &C, const int &BLOCK_SIZE) {
    assert(BLOCK_SIZE % WARP_SIZE == 0);
    assert(C < MAX_ALLOWED_C);

    const uint64 num_elements = x.numel();

    // we use vector instructions of width 4
    const int num_elements_per_block = BLOCK_SIZE << 2;
    const int NUM_BLOCKS = (num_elements + num_elements_per_block - 1) / num_elements_per_block;

    AT_DISPATCH_CUSTOM_INT_TYPES(x.scalar_type(), "contiguous_count_cuda_kernel", ([&] {
                                     _contiguous_count_cuda_kernel<scalar_t, 4><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                                         x.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), num_elements, C);
                                 }));
}
