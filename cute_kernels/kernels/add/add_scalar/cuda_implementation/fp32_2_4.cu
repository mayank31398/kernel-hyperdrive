#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../../../../include/dtypes/all.h"
#include "../../../../include/threads.h"

template <typename vector_t>
__global__ void _add_scalar_forward_cuda_kernel(const fp32 *x,
                                                const fp32 y,
                                                fp32 *output,
                                                const int64_t num_elements) {
    constexpr int vector_instruction_width = sizeof(vector_t) >> 2;
    static_assert(vector_instruction_width == 2 || vector_instruction_width == 4);

    const uint64 thread_id = get_global_thread_id();
    uint64 end = (thread_id + 1) * vector_instruction_width - 1;  // inclusive of last element

    if (end < num_elements) {
        vector_t *output_vec = (vector_t *)output;

        const fp32 *x_vec = (fp32 *)&((vector_t *)x)[thread_id];
        fp32 output_buffer[vector_instruction_width];

        // clang-format off
        #pragma unroll
        // clang-format on
        for (int i = 0; i < vector_instruction_width; i++) {
            output_buffer[i] = x_vec[i] + y;
        }

        if constexpr (vector_instruction_width == 2) {
            output_vec[thread_id] = DType<fp32>::make2(output_buffer);
        } else if constexpr (vector_instruction_width == 4) {
            output_vec[thread_id] = DType<fp32>::make4(output_buffer);
        } else {
            static_assert("vector_instruction_width is invalid for fp32");
        }
    }

    // use first warp for computing the last elements
    if (thread_id < WARP_SIZE) {
        // NOTE end is same as start since we don't use vector load stores here
        end = (num_elements / vector_instruction_width) * vector_instruction_width + thread_id;
        if (end < num_elements) {
            output[end] = x[end] + y;
        }
    }
}
