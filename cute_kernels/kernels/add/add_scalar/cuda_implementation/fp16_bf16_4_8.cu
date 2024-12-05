#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../../../../include/dtypes/all.h"
#include "../../../../include/threads.h"

template <typename scalar_t, typename vector_t>
__global__ void _add_scalar_forward_cuda_kernel_fp16_bf16_4_8(const scalar_t *x,
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

        const fp32 *x_vec = (fp32 *)&((vector_t *)x)[thread_id];

        constexpr int n = vector_instruction_width >> 1;
        fp32 output_buffer[n];

        // clang-format off
        #pragma unroll
        // clang-format on
        for (int i = 0; i < n; i++) {
            fp32_2 _x_upcast = dtype::upcast(dtype::reinterpret_32_bits_as_2x16(x_vec[i]));
            _x_upcast = DType<fp32>::make2(_x_upcast.x + y, _x_upcast.y + y);
            output_buffer[i] = dtype::reinterpret_2x16_as_32_bits(dtype::downcast(_x_upcast));
        }

        if constexpr (vector_instruction_width == 4) {
            output_vec[thread_id] = DType<fp32>::make2(output_buffer);
        } else if constexpr (vector_instruction_width == 8) {
            output_vec[thread_id] = DType<fp32>::make4(output_buffer);
        } else {
            static_assert("vector_instruction_width is invalid for fp16 & bf16");
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
