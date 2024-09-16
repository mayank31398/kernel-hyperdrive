#include "../../utils/dtypes.h"
#include "../../utils/threads.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void _vector_addition_forward_cuda_kernel(const scalar_t *x,
                                                     const scalar_t *y,
                                                     scalar_t *output,
                                                     const int num_elements) {
    const int thread_id = get_global_thread_id();
    const int num_elements_per_thread = get_num_elements_in_vector_dtype<scalar_t, fp32_4>();

    const int start = thread_id * num_elements_per_thread;
    const int end = (thread_id + 1) * num_elements_per_thread - 1; // inclusive of last element

    if (start < num_elements && end < num_elements) {
        // fp32_4 is a datatype used for vectorized loads and stores
        const fp32_4 *x4 = (const fp32_4 *)x;
        const fp32_4 *y4 = (const fp32_4 *)y;
        fp32_4 *output4 = (fp32_4 *)output;

        const fp32 *_x = (fp32 *)(&x4[thread_id]);
        const fp32 *_y = (fp32 *)(&y4[thread_id]);

        // tmp is initialized here to avoid doing multiple writes
        fp32_4 tmp4;
        fp32 *tmp = (fp32 *)(&tmp4);

        // clang-format off
        #pragma unroll
        // clang-format on
        for (int i = 0; i < 4; i++) {
            if (std::is_same_v<scalar_t, fp32>) {
                tmp[i] = _x[i] + _y[i];
            } else if constexpr (std::is_same_v<scalar_t, c10::Half> || std::is_same_v<scalar_t, c10::BFloat16>) {
                using dtype = DType<scalar_t>;
                using T2 = typename dtype::nv_dtype2;

                T2 x1 = dtype::unpack_from_fp32(_x[i]);
                T2 y1 = dtype::unpack_from_fp32(_y[i]);
                x1 = __hadd2(x1, y1);

                tmp[i] = dtype::pack_to_fp32(x1);
            } else {
                assert(false && "Function not implemented");
            }
        }

        output4[thread_id] = tmp4;
    } else if (start < num_elements) {
        // clang-format off
        #pragma unroll
        // clang-format on
        for (int i = start; i < num_elements; i++) {
            output[i] = x[i] + y[i];
        }
    }
}

void vector_addition_forward_cuda_kernel(
    torch::Tensor x, torch::Tensor y, torch::Tensor output, const int num_elements, const int BLOCK_SIZE) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "vector_addition_forward_cuda_kernel", ([&] {
            const int num_elements_per_thread = get_num_elements_in_vector_dtype<scalar_t, fp32_4>();

            const int num_elements_per_block = BLOCK_SIZE * num_elements_per_thread;
            const int NUM_BLOCKS = (num_elements + num_elements_per_block - 1) / num_elements_per_block;

            _vector_addition_forward_cuda_kernel<scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), num_elements);
        }));
}
