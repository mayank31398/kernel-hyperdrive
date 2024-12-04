#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../../../../include/dtypes/all.h"
#include "../../../../include/threads.h"

template <typename scalar_t, typename vector_t>
__global__ void _add_scalar_forward_cuda_kernel(const scalar_t *x,
                                                const fp32 y,
                                                scalar_t *output,
                                                const int64_t num_elements) {
    constexpr int vector_instruction_width = sizeof(vector_t) / sizeof(scalar_t);
    static_assert(vector_instruction_width == 2);

    using dtype = DType<scalar_t>;
    using T2 = typename dtype::nv_dtype2;

    const uint64 thread_id = get_global_thread_id();
    uint64 end = (thread_id + 1) * vector_instruction_width - 1;  // inclusive of last element

    if (end < num_elements) {
        vector_t *output_vec = (vector_t *)output;
        const fp64 *x_vec = (fp64 *)&((vector_t *)x)[thread_id];

        constexpr int n = vector_instruction_width >> 2;
        fp64 output_buffer[n];

        // clang-format off
        #pragma unroll
        // clang-format on
        for (int i = 0; i < n; i++) {
            auto [left, right] = dtype::reinterpret_64_bits_as_4x16(x_vec[i]);

            fp32_2 left_upcast = dtype::upcast(left);
            fp32_2 right_upcast = dtype::upcast(right);

            left_upcast = DType<fp32>::make2(left_upcast.x + y, left_upcast.y + y);
            right_upcast = DType<fp32>::make2(right_upcast.x + y, right_upcast.y + y);

            left = dtype::downcast(left_upcast);
            right = dtype::downcast(right_upcast);

            output_buffer[i] = dtype::reinterpret_4x16_as_64_bits(left, right);
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
