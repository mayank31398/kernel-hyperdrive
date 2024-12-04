#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../../../../include/dtypes/all.h"
#include "../../../../include/threads.h"

__global__ void _add_scalar_forward_cuda_kernel(const fp32 *x,
                                                const fp32 y,
                                                fp32 *output,
                                                const int64_t num_elements) {
    const int vector_instruction_width = 8;

    const uint64 thread_id = get_global_thread_id();
    uint64 end = (thread_id + 1) * vector_instruction_width - 1;  // inclusive of last element

    if (end < num_elements) {
        const fp64 *x_vec = (fp64 *)&((fp64_4 *)x)[thread_id];
        fp64 output_buffer[4];

        // clang-format off
        #pragma unroll
        // clang-format on
        for (int i = 0; i < n; i++) {
            fp32_2 _x_upcast = dtype::reinterpret_64_bits_as_2x32(x_vec[i]);
            output_buffer[i] = dtype::reinterpret_2x32_as_64_bits(_x_upcast.x + y, _x_upcast.y + y);
        }

        output_vec[thread_id] = DType<fp64>::make4(output_buffer);
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
