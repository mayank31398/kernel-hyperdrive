#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../../../include/dtypes/all.h"
#include "../../../include/threads.h"

#define MAX_ALLOWED_C 16384

inline __device__ void _initialize_shared_memory_and_output(
    uint32 *output_shared, int32 *output, const int &C, const int &thread_id, const int &num_loops_C) {
    // clang-format off
    #pragma unroll
    // clang-format on
    for (int i = 0; i < num_loops_C; i++) {
        const int index = thread_id + i * blockDim.x;
        if (index < C) {
            output_shared[index] = 0;
            output[index] = 0;
        }
    }
}

template <int vector_instruction_width>
__global__ void _contiguous_count_cuda_kernel(const int32 *x,
                                              int32 *output,
                                              const uint64 num_elements,
                                              const uint32 C) {
    const uint64 thread_id = get_global_thread_id();
    const int num_loops_C = (C + blockDim.x - 1) / blockDim.x;

    __shared__ uint32 output_shared[MAX_ALLOWED_C];
    _initialize_shared_memory_and_output(output_shared, output, C, thread_id, num_loops_C);
    __syncthreads();

    const int num_elements_per_SM = (num_elements + gridDim.x - 1) / gridDim.x;
    const int num_elements_per_loop = blockDim.x * vector_instruction_width;
    const int num_loops_B = (num_elements_per_SM + num_elements_per_loop - 1) / num_elements_per_loop;

    const int start = blockIdx.x * num_elements_per_SM;

    for (int i = 0; i < num_loops_B; i++) {
        int index = thread_id + i * num_loops_B + start;
        for (int j = 0; j < vector_instruction_width; j++) {
            // x_vec[index + j]
        }
    }

    // // TODO add code here

    // __syncthreads();

    // // clang-format off
    // #pragma unroll
    // // clang-format on
    // for (int i = 0; i < num_loops; i++) {
    //     const int index = thread_id + i * blockDim.x;
    //     if (index < C) {
    //         atomicAdd(&output[index], output_shared[index]);
    //     }
    // }
}

void contiguous_count_cuda(const torch::Tensor &x, const torch::Tensor &output, const int &C, const int &BLOCK_SIZE) {
    assert(BLOCK_SIZE % WARP_SIZE == 0);
    assert(C < MAX_ALLOWED_C);

    const uint64 num_elements = x.numel();

    // we use vector instructions of width 4
    const int num_elements_per_block = BLOCK_SIZE << 2;
    const int NUM_BLOCKS = (num_elements + num_elements_per_block - 1) / num_elements_per_block;

    _contiguous_count_cuda_kernel<4>
        <<<NUM_BLOCKS, BLOCK_SIZE>>>(x.data_ptr<int32>(), output.data_ptr<int32>(), num_elements, C);
}
