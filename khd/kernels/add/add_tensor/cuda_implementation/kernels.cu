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
}

torch::Tensor add_tensor_forward_cuda_kernel_dispatch(const torch::Tensor x,
                                                      const torch::Tensor y,
                                                      const int &vectorized_load_store_size,
                                                      const int &BLOCK_SIZE) {
    torch::Tensor output = torch::empty_like(x);
    const int num_elements = output.numel();

    using kernel_t = void (*)(scalar_t *, scalar_t *, scalar_t *, int, int);
    auto kernel_func = static_cast<kernel_t>(nullptr);

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(
        x.scalar_type(), "add_tensor_forward_cuda_kernel", ([&] {
        const int num_elements_per_block = BLOCK_SIZE * vectorized_load_store_size;
        const int NUM_BLOCKS = (num_elements + num_elements_per_block - 1) / num_elements_per_block;

        switch (vectorized_load_store_size) {
        case 1:
            kernel_func = _add_tensor_forward_cuda_kernel<scalar_t, scalar_t>;
            break;
        case 2: {
            using vector_t = typename DType<scalar_t>::nv_dtype2;
            kernel_func = _add_tensor_forward_cuda_kernel<scalar_t, vector_t>;
            break;
        }
        case 4:
            if (std::is_same_v<scalar_t, fp32>) {
                kernel_func = _add_tensor_forward_cuda_kernel<scalar_t, fp32_4>;
            } else {
                kernel_func = _add_tensor_forward_cuda_kernel<scalar_t, fp32_2>;
            }
            break;
        case 8:
            if (std::is_same_v<scalar_t, fp32>) {
                assert(false);
            } else {
                kernel_func = _add_tensor_forward_cuda_kernel<scalar_t, fp32_4>;
            }
            break;
        default:
            assert(false);
        }

        kernel_func<<<NUM_BLOCKS, BLOCK_SIZE>>>(x.data_ptr<scalar_t>(),
                                                y.data_ptr<scalar_t>(),
                                                output.data_ptr<scalar_t>(),
                                                num_elements,
                                                vectorized_load_store_size);
));

return output;
}
