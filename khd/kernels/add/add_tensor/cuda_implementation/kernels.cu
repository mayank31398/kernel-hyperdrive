#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../../../utils/dtypes.h"
#include "../../../utils/threads.h"

template <typename scalar_t>
__global__ void _add_tensor_forward_cuda_kernel(const scalar_t *x,
                                                const scalar_t *y,
                                                scalar_t *output,
                                                const int64_t num_elements) {
    const int64_t thread_id = get_global_thread_id();
    const int vector_instruction_width = sizeof(fp32_4) / sizeof(scalar_t);

    const int64_t start = thread_id * vector_instruction_width;
    const int64_t end = (thread_id + 1) * vector_instruction_width - 1;  // inclusive of last element

    if (start < num_elements && end < num_elements) {
        const fp32 *_x = (fp32 *)&((fp32_4 *)x)[thread_id];
        const fp32 *_y = (fp32 *)&((fp32_4 *)y)[thread_id];

        fp32 output_buffer[4];

        // clang-format off
        #pragma unroll
        // clang-format on
        for (int i = 0; i < 4; i++) {
            if constexpr (std::is_same_v<scalar_t, fp32>) {
                output_buffer[i] = _x[i] + _y[i];
            } else {
                using dtype = DType<scalar_t>;
                output_buffer[i] = dtype::reinterpret_2x16_as_32_bits(
                    __hadd2(dtype::reinterpret_32_bits_as_2x16(_x[i]), dtype::reinterpret_32_bits_as_2x16(_y[i])));
            }
        }

        ((fp32_4 *)output)[thread_id] = DType<fp32>::make4(output_buffer);
    } else if (start < num_elements) {
        // clang-format off
        #pragma unroll
        // clang-format on
        for (int64_t i = start; i < num_elements; i++) {
            output[i] = x[i] + y[i];
        }
    }
}

void add_tensor_forward_cuda(const torch::Tensor x,
                             const torch::Tensor y,
                             torch::Tensor output,
                             const int &BLOCK_SIZE) {
    const int64_t num_elements = x.numel();

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(
        x.scalar_type(), "add_tensor_forward_cuda_kernel", ([&] {
            const int vector_instruction_width = sizeof(fp32_4) / sizeof(scalar_t);

            const int num_elements_per_block = BLOCK_SIZE * vector_instruction_width;
            const int NUM_BLOCKS = (num_elements + num_elements_per_block - 1) / num_elements_per_block;

            _add_tensor_forward_cuda_kernel<scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), num_elements);
        }));
}
