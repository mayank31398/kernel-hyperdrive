#include "../../../utils/dtypes.h"
#include "../../../utils/threads.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename scalar_t, typename vector_t>
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
        const int start = thread_id * vectorized_load_store_size;
        const int end = (thread_id + 1) * vectorized_load_store_size - 1; // inclusive of last element

        if (start < num_elements && end < num_elements) {
            if (std::is_same_v<scalar_t, fp32>) {
                fp32 *output_buffer = new fp32[vectorized_load_store_size];

                // clang-format off
                #pragma unroll
                // clang-format on
                for (int i = 0; i < vectorized_load_store_size; i++) {
                    output_buffer[i] = ((vector_t *)x)[i] + ((vector_t *)y)[i];
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
}

torch::Tensor add_tensor_forward_cuda_kernel_dispatch(const torch::Tensor x,
                                                      const torch::Tensor y,
                                                      const int &vectorized_load_store_size,
                                                      const int &BLOCK_SIZE) {
    torch::Tensor output = torch::empty_like(x);
    const int num_elements = output.numel();

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(x.scalar_type(), "add_tensor_forward_cuda_kernel", ([&] {
                                       const int num_elements_per_block = BLOCK_SIZE * vectorized_load_store_size;
                                       const int NUM_BLOCKS =
                                           (num_elements + num_elements_per_block - 1) / num_elements_per_block;

                                       _add_tensor_forward_cuda_kernel<scalar_t, scalar_t>
                                           <<<NUM_BLOCKS, BLOCK_SIZE>>>(x.view(-1).data_ptr<scalar_t>(),
                                                                        y.view(-1).data_ptr<scalar_t>(),
                                                                        output.view(-1).data_ptr<scalar_t>(),
                                                                        num_elements,
                                                                        1);

                                       // switch (vectorized_load_store_size) {
                                       // case 1:
                                       //     _add_tensor_forward_cuda_kernel<scalar_t, scalar_t>
                                       //         <<<NUM_BLOCKS, BLOCK_SIZE>>>(x.data_ptr<scalar_t>(),
                                       //                                      y.data_ptr<scalar_t>(),
                                       //                                      output.data_ptr<scalar_t>(),
                                       //                                      num_elements,
                                       //                                      vectorized_load_store_size);
                                       //     break;
                                       // case 2: {
                                       //     using vector_t = typename DType<scalar_t>::nv_dtype2;
                                       //     _add_tensor_forward_cuda_kernel<scalar_t, vector_t>
                                       //         <<<NUM_BLOCKS, BLOCK_SIZE>>>(x.data_ptr<scalar_t>(),
                                       //                                      y.data_ptr<scalar_t>(),
                                       //                                      output.data_ptr<scalar_t>(),
                                       //                                      num_elements,
                                       //                                      vectorized_load_store_size);
                                       //     break;
                                       // }
                                       // case 4:
                                       //     if constexpr (std::is_same_v<scalar_t, fp32>) {
                                       //         _add_tensor_forward_cuda_kernel<fp32, fp32_4>
                                       //             <<<NUM_BLOCKS, BLOCK_SIZE>>>(x.data_ptr<scalar_t>(),
                                       //                                          y.data_ptr<scalar_t>(),
                                       //                                          output.data_ptr<scalar_t>(),
                                       //                                          num_elements,
                                       //                                          vectorized_load_store_size);
                                       //     } else {
                                       //         _add_tensor_forward_cuda_kernel<scalar_t, fp32_2>
                                       //             <<<NUM_BLOCKS, BLOCK_SIZE>>>(x.data_ptr<scalar_t>(),
                                       //                                          y.data_ptr<scalar_t>(),
                                       //                                          output.data_ptr<scalar_t>(),
                                       //                                          num_elements,
                                       //                                          vectorized_load_store_size);
                                       //     }
                                       //     break;
                                       // default:
                                       //     _add_tensor_forward_cuda_kernel<scalar_t, fp32_4>
                                       //         <<<NUM_BLOCKS, BLOCK_SIZE>>>(x.data_ptr<scalar_t>(),
                                       //                                      y.data_ptr<scalar_t>(),
                                       //                                      output.data_ptr<scalar_t>(),
                                       //                                      num_elements,
                                       //                                      vectorized_load_store_size);
                                       //     break;
                                       // }

                                       // if (!kernel_func) {
                                       //     throw std::runtime_error("Kernel function is not set correctly");
                                       // }

                                       // cudaError_t err = cudaGetLastError();
                                       // if (err != cudaSuccess) {
                                       //     throw std::runtime_error("Kernel launch failed: " +
                                       //     std::string(cudaGetErrorString(err)));
                                       // }
                                   }));

    return output;
}
