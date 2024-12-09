#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../../../../include/dtypes/all.h"
#include "../../../../include/threads.h"

template <typename scalar_t, typename vector_t>
__global__ void _add_tensor_forward_cuda_kernel(const scalar_t *x,
                                                const scalar_t *y,
                                                scalar_t *output,
                                                const int64_t num_elements) {
    constexpr int vector_instruction_width = sizeof(vector_t) / sizeof(scalar_t);
    static_assert(vector_instruction_width == 1 || vector_instruction_width == 2 || vector_instruction_width == 4 ||
                  vector_instruction_width == 8 || vector_instruction_width == 16);

    using dtype = DType<scalar_t>;
    using T = typename dtype::nv_dtype;
    using T2 = typename dtype::nv_dtype2;

    const uint64 thread_id = get_global_thread_id();

    if constexpr (vector_instruction_width == 1) {
        if (thread_id < num_elements) {
            output[thread_id] = x[thread_id] + y[thread_id];
        }
    } else {
        uint64 end = (thread_id + 1) * vector_instruction_width - 1;  // inclusive of last element

        if (end < num_elements) {
            vector_t *output_vec = (vector_t *)output;

            if constexpr (std::is_same_v<scalar_t, fp32>) {
                if constexpr (vector_instruction_width == 8) {
                    const fp64 *x_vec = (fp64 *)&((vector_t *)x)[thread_id];
                    const fp64 *y_vec = (fp64 *)&((vector_t *)y)[thread_id];

                    constexpr int n = vector_instruction_width >> 1;
                    fp64 output_buffer[n];

                    // clang-format off
                    #pragma unroll
                    // clang-format on
                    for (int i = 0; i < n; i++) {
                        T2 _x = dtype::reinterpret_64_bits_as_2x32(x_vec[i]);
                        T2 _y = dtype::reinterpret_64_bits_as_2x32(y_vec[i]);

                        output_buffer[i] = dtype::reinterpret_2x32_as_64_bits(_x.x + _y.x, _x.y + _y.y);
                    }

                    output_vec[thread_id] = DType<fp64>::make4(output_buffer);
                } else {
                    const fp32 *x_vec = (fp32 *)&((vector_t *)x)[thread_id];
                    const fp32 *y_vec = (fp32 *)&((vector_t *)y)[thread_id];
                    fp32 output_buffer[vector_instruction_width];

                    // clang-format off
                    #pragma unroll
                    // clang-format on
                    for (int i = 0; i < vector_instruction_width; i++) {
                        output_buffer[i] = x_vec[i] + y_vec[i];
                    }

                    if constexpr (vector_instruction_width == 2) {
                        output_vec[thread_id] = dtype::make2(output_buffer);
                    } else if constexpr (vector_instruction_width == 4) {
                        output_vec[thread_id] = dtype::make4(output_buffer);
                    } else {
                        static_assert("vector_instruction_width is invalid for fp32");
                    }
                }
            } else {
                if constexpr (vector_instruction_width == 2) {
                    const T2 _x = ((vector_t *)x)[thread_id];
                    const T2 _y = ((vector_t *)y)[thread_id];

                    output_vec[thread_id] = __hadd2(_x, _y);
                } else if (vector_instruction_width == 16) {
                    const fp64 *x_vec = (fp64 *)&((vector_t *)x)[thread_id];
                    const fp64 *y_vec = (fp64 *)&((vector_t *)y)[thread_id];

                    constexpr int n = vector_instruction_width >> 2;
                    fp64 output_buffer[n];

                    // clang-format off
                    #pragma unroll
                    // clang-format on
                    for (int i = 0; i < n; i++) {
                        auto [x_first, x_second, x_third, x_fourth] = dtype::reinterpret_64_bits_as_4x16(x_vec[i]);
                        auto [y_first, y_second, y_third, y_fourth] = dtype::reinterpret_64_bits_as_4x16(y_vec[i]);

                        T2 x_left = dtype::make2(x_first, x_second);
                        T2 y_left = dtype::make2(y_first, y_second);
                        x_left = __hadd2(x_left, y_left);

                        T2 x_right = dtype::make2(x_third, x_fourth);
                        T2 y_right = dtype::make2(y_third, y_fourth);
                        x_right = __hadd2(x_right, y_right);

                        output_buffer[i] = dtype::reinterpret_4x16_as_64_bits(x_left, x_right);
                    }

                    output_vec[thread_id] = DType<fp64>::make4(output_buffer);
                } else {
                    const fp32 *x_vec = (fp32 *)&((vector_t *)x)[thread_id];
                    const fp32 *y_vec = (fp32 *)&((vector_t *)y)[thread_id];

                    constexpr int n = vector_instruction_width >> 1;
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

                    if constexpr (vector_instruction_width == 4) {
                        output_vec[thread_id] = DType<fp32>::make2(output_buffer);
                    } else if constexpr (vector_instruction_width == 8) {
                        output_vec[thread_id] = DType<fp32>::make4(output_buffer);
                    } else {
                        static_assert("vector_instruction_width is invalid for fp16 & bf16");
                    }
                }
            }
        }

        // use first warp for computing the last elements
        if (thread_id < WARP_SIZE) {
            // NOTE end is same as start since we don't use vector load stores here
            end = (num_elements / vector_instruction_width) * vector_instruction_width + thread_id;
            if (end < num_elements) {
                output[end] = x[end] + y[end];
            }
        }
    }
}

void add_tensor_forward_cuda(const torch::Tensor &x,
                             const torch::Tensor &y,
                             torch::Tensor &output,
                             const int &vector_instruction_width,
                             const int &BLOCK_SIZE) {
    assert(BLOCK_SIZE % WARP_SIZE == 0);

    const int64_t num_elements = x.numel();

    const int num_elements_per_block = BLOCK_SIZE * vector_instruction_width;
    const int NUM_BLOCKS = (num_elements + num_elements_per_block - 1) / num_elements_per_block;

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(
        x.scalar_type(), "add_tensor_forward_cuda_kernel", ([&] {
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
                        _add_tensor_forward_cuda_kernel<scalar_t, fp64_4><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                            x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), num_elements);
                    } else {
                        _add_tensor_forward_cuda_kernel<scalar_t, fp32_4><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                            x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), num_elements);
                    }
                    break;
                case 16:
                    if constexpr (std::is_same_v<scalar_t, c10::Half> || std::is_same_v<scalar_t, c10::BFloat16>) {
                        _add_tensor_forward_cuda_kernel<scalar_t, fp64_4><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                            x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), num_elements);
                    } else {
                        throw std::runtime_error("invalid vector_instruction_width = 16 for fp32");
                    }
                    break;
                default:
                    throw std::runtime_error("invalid vector_instruction_width");
                    break;
            }
        }));
}
