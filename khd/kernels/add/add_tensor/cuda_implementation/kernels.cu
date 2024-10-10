#include "../../../utils/dtypes.h"
#include "../../../utils/threads.h"
#include "../../../utils/vector_dtypes.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void _add_tensor_forward_cuda_kernel(const scalar_t *x,
                                                const scalar_t *y,
                                                scalar_t *output,
                                                const int num_elements,
                                                const int vectorized_load_store_size) {
    const int thread_id = get_global_thread_id();

    if (vectorized_load_store_size == 1) {
        if (thread_id < num_elements) {
            output[thread_id] = x[thread_id] + y[thread_id];
        }
    } else {
        using vector_t = VectorDTypeSelector<vectorized_load_store_size, scalar_t>::vector_t;

        const int start = thread_id * vectorized_load_store_size;
        const int end = (thread_id + 1) * vectorized_load_store_size - 1; // inclusive of last element

        if (start < num_elements && end < num_elements) {
            if (std::is_same_v<scalar_t, fp32>) {
                assert(vectorized_load_store_size == 2 || vectorized_load_store_size == 4);

                fp32 output_buffer[vectorized_load_store_size];

                // clang-format off
                #pragma unroll
                // clang-format on
                for (int i = 0; i < vectorized_load_store_size; i++) {
                    output_buffer[i] = reinterpret_cast<vector_t *>(x)[i] + reinterpret_cast<vector_t *>(y)[i];
                }
            }
        } else if (start < num_elements) {
            // clang-format off
            #pragma unroll
            // clang-format on
            for (int i = start; i < num_elements; i++) {
                output[i] = x[i] + y[i];
            }
        }
    }

    // using dtype = DType<scalar_t>;
    // using T2 = typename dtype::nv_dtype2;

    // if (start < num_elements && end < num_elements) {
    //     const fp32 *x_vec = (fp32 *)&((const vecT *)x)[thread_id];
    //     const fp32 *y_vec = (fp32 *)&((const vecT *)y)[thread_id];

    //     fp32 output_buffer[vectorized_load_store_size];

    //     // clang-format off
    //     #pragma unroll
    //     // clang-format on
    //     for (int i = 0; i < vectorized_load_store_size; i++) {
    //         if (std::is_same_v<scalar_t, fp32>) {
    //             output_buffer[i] = x_vec[i] + y_vec[i];
    //         } else if constexpr (std::is_same_v<scalar_t, c10::Half> ||
    //         std::is_same_v<scalar_t, c10::BFloat16>) {
    //             T2 _x = dtype::reinterpret_32_bits_as_2x16(x_vec[i]);
    //             T2 _y = dtype::reinterpret_32_bits_as_2x16(y_vec[i]);
    //             _x = __hadd2(_x, _y);

    //             output_buffer[i] = dtype::reinterpret_2x16_as_32_bits(_x);
    //         } else {
    //             assert(false && "Function not implemented");
    //         }
    //     }

    //     if (vectorized_load_store_size == 1) {
    //         output[thread_id] =
    //     }
    //     ((vecT *)output)[thread_id] =
    //         make_float4(output_buffer[0], output_buffer[1], output_buffer[2],
    //         output_buffer[3]);
    // } else if (start < num_elements) {
    //     // clang-format off
    //     #pragma unroll
    //     // clang-format on
    //     for (int i = start; i < num_elements; i++) {
    //         output[i] = x[i] + y[i];
    //     }
    // }
}

torch::Tensor add_tensor_forward_cuda_kernel_dispatch(const torch::Tensor x,
                                                      const torch::Tensor y,
                                                      const int &vectorized_load_store_size,
                                                      const int &BLOCK_SIZE) {
    torch::Tensor output = torch::empty_like(x);
    const int num_elements = output.numel();

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(
        x.scalar_type(), "add_tensor_forward_cuda_kernel", ([&] {
            const int num_elements_per_block = BLOCK_SIZE * vectorized_load_store_size;
            const int NUM_BLOCKS = (num_elements + num_elements_per_block - 1) / num_elements_per_block;

            _add_tensor_forward_cuda_kernel<scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(x.data_ptr<scalar_t>(),
                                                                                  y.data_ptr<scalar_t>(),
                                                                                  output.data_ptr<scalar_t>(),
                                                                                  num_elements,
                                                                                  vectorized_load_store_size);
        }));

    return output;
}
