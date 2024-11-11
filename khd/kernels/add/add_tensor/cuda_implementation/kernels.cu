#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../../../utils/dtypes.h"
#include "../../../utils/threads.h"

template <typename scalar_t, typename vector_t>
__global__ void _add_tensor_forward_cuda_kernel(const scalar_t *x,
                                                const scalar_t *y,
                                                scalar_t *output,
                                                const int64_t num_elements) {
    const int vector_instruction_width = sizeof(vector_t) / sizeof(scalar_t);
    const int64_t thread_id = get_global_thread_id();

    // no vector instructions
    if constexpr (vector_instruction_width == 1) {
        if (thread_id < num_elements) {
            output[thread_id] = x[thread_id] + y[thread_id];
        }
    } else {
        // no vector instructions
        using dtype = DType<scalar_t>;

        const int64_t start = thread_id * vector_instruction_width;
        const int64_t end = (thread_id + 1) * vector_instruction_width - 1;  // inclusive of last element

        if (start < num_elements && end < num_elements) {
            vector_t *output_vec = (vector_t *)output;

            if constexpr (std::is_same_v<scalar_t, fp32>) {
                const fp32 *x_vec = (fp32 *)&((vector_t *)x)[thread_id];
                const fp32 *y_vec = (fp32 *)&((vector_t *)y)[thread_id];
                fp32 output_buffer[vector_instruction_width];

                // clang-format off
                #pragma unroll
                // clang-format on
                for (int i = 0; i < vector_instruction_width; i++) {
                    output_buffer[i] = x_vec[i] + y_vec[i];
                }

                if constexpr (std::is_same_v<vector_t, fp32_2>) {
                    static_assert(vector_instruction_width == 2);
                    output_vec[thread_id] = dtype::make2(output_buffer);
                } else if constexpr (std::is_same_v<vector_t, fp32_4>) {
                    static_assert(vector_instruction_width == 4);
                    output_vec[thread_id] = dtype::make4(output_buffer);
                }
            } else {
                using T2 = DType<scalar_t>;

                if constexpr (std::is_same_v<vector_t, fp16_2> || std::is_same_v<vector_t, bf16_2>) {
                    T2 _x = ((vector_t *)x)[thread_id];
                    T2 _y = ((vector_t *)y)[thread_id];

                    output_vec[thread_id] = __hadd2(_x, _y);
                } else {
                    const fp32 *x_vec = (fp32 *)&((vector_t *)x)[thread_id];
                    const fp32 *y_vec = (fp32 *)&((vector_t *)y)[thread_id];

                    const int n = vector_instruction_width >> 1;
                    fp32 output_buffer[n];

                    // clang-format off
                    #pragma unroll
                    // clang-format on
                    for (int i = 0; i < n; i++) {
                        T2 _x = dtype::reinterpret_32_bits_as_2x16(x_vec[i]);
                        T2 _y = dtype::reinterpret_32_bits_as_2x16(y_vec[i]);

                        _x = __hadd2(_x, _y);
                        output_buffer[i] = dtype::reinterpret_2x16_as_32_bits(_x);
                    }

                    if constexpr (std::is_same_v<vector_t, fp32_2>) {
                        assert(vector_instruction_width == 4);
                        output_vec[thread_id] = DType<fp32>::make2(output_buffer);
                    } else if constexpr (std::is_same_v<vector_t, fp32_4>) {
                        assert(vector_instruction_width == 8);
                        output_vec[thread_id] = DType<fp32>::make4(output_buffer);
                    }
                }
            }
        } else if (start < num_elements) {
            // clang-format off
            #pragma unroll
            // clang-format on
            for (int64_t i = start; i < num_elements; i++) {
                output[i] = x[i] + y[i];
            }
        }
    }
}

void add_tensor_forward_cuda(const torch::Tensor x,
                             const torch::Tensor y,
                             torch::Tensor output,
                             const int &vector_instruction_width,
                             const int &BLOCK_SIZE) {
    const int64_t num_elements = x.numel();

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(
        x.scalar_type(), "add_tensor_forward_cuda_kernel", ([&] {
            const int num_elements_per_block = BLOCK_SIZE * vector_instruction_width;
            const int NUM_BLOCKS = (num_elements + num_elements_per_block - 1) / num_elements_per_block;

            switch (vector_instruction_width) {
                case 1:
                    _add_tensor_forward_cuda_kernel<scalar_t, scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                        x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), num_elements);
                    break;
                case 2:
                    using vector_t = typename DType<scalar_t>::nv_dtype2;
                    _add_tensor_forward_cuda_kernel<scalar_t, vector_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                        x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), num_elements);
                    break;
                case 4:
                    if constexpr (std::is_same_v<scalar_t, fp32>) {
                        _add_tensor_forward_cuda_kernel<scalar_t, fp32_4><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                            x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), num_elements);
                    } else {
                        _add_tensor_forward_cuda_kernel<scalar_t, fp32_2><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                            x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), num_elements);
                    }
                    break;
                case 8:
                    if constexpr (std::is_same_v<scalar_t, fp32>) {
                        throw std::runtime_error("fp32 doesn't support vector_instruction_width = 8");
                    } else {
                        _add_tensor_forward_cuda_kernel<scalar_t, fp32_4><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                            x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), num_elements);
                    }
                    break;
                default:
                    throw std::runtime_error("invalid vector_instruction_width");
                    break;
            }
        }));
}
