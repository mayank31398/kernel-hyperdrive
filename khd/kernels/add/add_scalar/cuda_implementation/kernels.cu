#include "../../../utils/dtypes.h"
#include "../../../utils/threads.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void _add_scalar_forward_cuda_kernel(const scalar_t *x,
                                                const fp32 y,
                                                scalar_t *output,
                                                const int num_elements) {
    const int thread_id = get_global_thread_id();
    const int num_elements_per_thread = get_num_elements_in_vector_dtype<scalar_t, fp32_4>();

    const int start = thread_id * num_elements_per_thread;
    const int end = (thread_id + 1) * num_elements_per_thread - 1; // inclusive of last element

    using dtype = DType<scalar_t>;
    using T2 = typename dtype::nv_dtype2;

    if (start < num_elements && end < num_elements) {
        const fp32 *x_vec = (fp32 *)&((const fp32_4 *)x)[thread_id];

        fp32 output_buffer[4];

        // clang-format off
        #pragma unroll
        // clang-format on
        for (int i = 0; i < 4; i++) {
            if (std::is_same_v<scalar_t, fp32>) {
                output_buffer[i] = x_vec[i] + y;
            } else if constexpr (std::is_same_v<scalar_t, c10::Half> || std::is_same_v<scalar_t, c10::BFloat16>) {
                T2 _x = dtype::reinterpret_32_bits_as_2x16(x_vec[i]);

                fp32_2 _x_upcast = dtype::upcast(_x);
                _x_upcast = make_float2(_x_upcast.x + y, _x_upcast.y + y);

                _x = dtype::downcast(_x_upcast);

                output_buffer[i] = dtype::reinterpret_2x16_as_32_bits(_x);
            } else {
                assert(false && "Function not implemented");
            }
        }

        ((fp32_4 *)output)[thread_id] =
            make_float4(output_buffer[0], output_buffer[1], output_buffer[2], output_buffer[3]);
    } else if (start < num_elements) {
        // clang-format off
        #pragma unroll
        // clang-format on
        for (int i = start; i < num_elements; i++) {
            output[i] = x[i] + y;
        }
    }
}

void add_scalar_forward_cuda_kernel(
    const torch::Tensor x, const float &y, torch::Tensor output, const int &num_elements, const int &BLOCK_SIZE) {
    AT_DISPATCH_CUSTOM_FLOAT_TYPES(
        x.scalar_type(), "add_scalar_forward_cuda_kernel", ([&] {
            const int num_elements_per_thread = get_num_elements_in_vector_dtype<scalar_t, fp32_4>();

            const int num_elements_per_block = BLOCK_SIZE * num_elements_per_thread;
            const int NUM_BLOCKS = (num_elements + num_elements_per_block - 1) / num_elements_per_block;

            _add_scalar_forward_cuda_kernel<scalar_t>
                <<<NUM_BLOCKS, BLOCK_SIZE>>>(x.data_ptr<scalar_t>(), y, output.data_ptr<scalar_t>(), num_elements);
        }));
}
