#include "../../dtypes.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define BLOCK_SIZE 1024

// for vectorized load store
std::unordered_map<std::type_index, int> num_elements_per_thread_mapping = {
    {typeid(fp32), 4}, {typeid(c10::Half), 8}, {typeid(c10::BFloat16), 8}};

template <typename scalar_t>
__global__ void vector_addition_forward_kernel(const scalar_t *x,
                                               const scalar_t *y,
                                               scalar_t *output,
                                               const int num_elements,
                                               const int num_elements_per_thread) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

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
        fp32_4 tmp;
        fp32 *_tmp = (fp32 *)(&tmp);

        if (std::is_same_v<scalar_t, fp32>) {
            // clang-format off
            #pragma unroll
            // clang-format on
            for (int i = 0; i < 4; i++) {
                _tmp[i] = _x[i] + _y[i];
            }
        } else if constexpr (std::is_same_v<scalar_t, c10::Half> || std::is_same_v<scalar_t, c10::BFloat16>) {
            DType<scalar_t> q;

            // clang-format off
            #pragma unroll
            // clang-format on
            for (int i = 0; i < 4; i++) {
                _tmp[i] = q.pack_to_fp32(__hadd2(q.unpack_from_fp32(_x[i]), q.unpack_from_fp32(_y[i])));
            }
        } else {
            assert(false && "Function not implemented");
        }

        output4[thread_id] = tmp;
    } else if (start < num_elements) {
#pragma unroll
        for (int i = start; i < num_elements; i++) {
            output[i] = x[i] + y[i];
        }
    }
}

torch::Tensor vector_addition_forward_kernel_dispatcher(torch::Tensor x, torch::Tensor y) {
    int num_elements = x.numel();

    torch::Tensor output = torch::empty_like(x);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "vector_addition_forward_kernel", ([&] {
            int num_elements_per_thread = num_elements_per_thread_mapping[std::type_index(typeid(scalar_t))];

            int num_elements_per_block = BLOCK_SIZE * num_elements_per_thread;
            int NUM_BLOCKS = (num_elements + num_elements_per_block - 1) / num_elements_per_block;

            vector_addition_forward_kernel<scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(x.data_ptr<scalar_t>(),
                                                                                 y.data_ptr<scalar_t>(),
                                                                                 output.data_ptr<scalar_t>(),
                                                                                 num_elements,
                                                                                 num_elements_per_thread);
        }));

    return output;
}
