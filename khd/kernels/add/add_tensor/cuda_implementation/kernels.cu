#include "../../../utils/dtypes.h"
#include "../../../utils/threads.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define NAIVE_KERNEL_NAME "add_tensor_forward_naive_cuda_kernel"
#define EFFICIENT_KERNEL_NAME "add_tensor_forward_efficient_cuda_kernel"

template <typename scalar_t>
__global__ void _add_tensor_forward_naive_cuda_kernel(const scalar_t *x,
                                                      const scalar_t *y,
                                                      scalar_t *output,
                                                      const int num_elements) {
    const int thread_id = get_global_thread_id();

    if (thread_id < num_elements) {
        output[thread_id] = x[thread_id] + y[thread_id];
    }
}

template <typename scalar_t>
__global__ void _add_tensor_forward_efficient_cuda_kernel(const scalar_t *x,
                                                          const scalar_t *y,
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
        const fp32 *y_vec = (fp32 *)&((const fp32_4 *)y)[thread_id];

        fp32 output_buffer[4];

        // clang-format off
        #pragma unroll
        // clang-format on
        for (int i = 0; i < 4; i++) {
            if (std::is_same_v<scalar_t, fp32>) {
                output_buffer[i] = x_vec[i] + y_vec[i];
            } else if constexpr (std::is_same_v<scalar_t, c10::Half> || std::is_same_v<scalar_t, c10::BFloat16>) {
                T2 _x = dtype::reinterpret_32_bits_as_2x16(x_vec[i]);
                T2 _y = dtype::reinterpret_32_bits_as_2x16(y_vec[i]);
                _x = __hadd2(_x, _y);

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
            output[i] = x[i] + y[i];
        }
    }
}

void add_tensor_forward_cuda_kernel_dispatch(const torch::Tensor x,
                                             const torch::Tensor y,
                                             torch::Tensor output,
                                             const bool &use_efficient_kernel,
                                             const int &num_elements,
                                             const int &BLOCK_SIZE) {
    str kernel_name;
    if (use_efficient_kernel) {
        kernel_name = EFFICIENT_KERNEL_NAME;
    } else {
        kernel_name = NAIVE_KERNEL_NAME;
    }

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(
        x.scalar_type(), kernel_name, ([&] {
            if (use_efficient_kernel) {
                const int num_elements_per_thread = get_num_elements_in_vector_dtype<scalar_t, fp32_4>();
                const int num_elements_per_block = BLOCK_SIZE * num_elements_per_thread;
                const int NUM_BLOCKS = (num_elements + num_elements_per_block - 1) / num_elements_per_block;

                _add_tensor_forward_efficient_cuda_kernel<scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                    x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), num_elements);
            } else {
                const int NUM_BLOCKS = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

                _add_tensor_forward_naive_cuda_kernel<scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                    x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), num_elements);
            }
        }));
}
