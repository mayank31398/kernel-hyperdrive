#include "../../utils/dtypes.h"
#include "../../utils/threads.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename scalar_t, typename T, typename vecT>
__global__ void _vector_addition_forward_cuda_kernel(const scalar_t *x,
                                                     const scalar_t *y,
                                                     scalar_t *output,
                                                     const int num_elements) {
    const int thread_id = get_global_thread_id();
    const int num_elements_per_thread = get_num_elements_in_vector_dtype<T, vecT>();

    const int start = thread_id * num_elements_per_thread;
    const int end = (thread_id + 1) * num_elements_per_thread - 1; // inclusive of last element

    if (start < num_elements && end < num_elements) {
        // fp32_4 is a datatype used for vectorized loads and stores
        const vecT *x_vec = (const vecT *)x;
        const vecT *y_vec = (const vecT *)y;
        vecT *output_vec = (vecT *)output;

        const fp32 *_x = (fp32 *)(&x_vec[thread_id]);
        const fp32 *_y = (fp32 *)(&y_vec[thread_id]);

        // tmp is initialized here to avoid doing multiple writes
        vecT tmp_vec;
        fp32 *tmp = (fp32 *)(&tmp_vec);

        // clang-format off
        #pragma unroll
        // clang-format on
        for (int i = 0; i < 4; i++) {
            if (std::is_same_v<T, fp32>) {
                tmp[i] = _x[i] + _y[i];
            } else if constexpr (std::is_same_v<T, fp16> || std::is_same_v<T, bf16>) {
                DType<T> q;
                tmp[i] = q.pack_to_fp32(__hadd2(q.unpack_from_fp32(_x[i]), q.unpack_from_fp32(_y[i])));
            } else {
                assert(false && "Function not implemented");
            }
        }

        output_vec[thread_id] = tmp_vec;
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
            using T = DType<scalar_t>::nv_dtype;
            using vecT = fp32_4;

            const int num_elements_per_thread = get_num_elements_in_vector_dtype<T, vecT>();

            const int num_elements_per_block = BLOCK_SIZE * num_elements_per_thread;
            const int NUM_BLOCKS = (num_elements + num_elements_per_block - 1) / num_elements_per_block;

            _vector_addition_forward_cuda_kernel<scalar_t, T, vecT><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), num_elements);
        }));
}
