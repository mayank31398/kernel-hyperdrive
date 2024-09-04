#include "../../dtypes.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define BLOCK_SIZE 1024

std::unordered_map<std::type_index, int> num_elements_per_thread_mapping = {
    {typeid(fp32), 4}, {typeid(c10::Half), 8}, {typeid(c10::BFloat16), 8}};

// for vectorized load store
#define NUM_ELEMENTS_PER_THREAD_FP32 4
#define NUM_ELEMENTS_PER_THREAD_FP16 8
#define NUM_ELEMENTS_PER_THREAD_BF16 8

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

        // tmp is initialized here to avoid doing multiple writes
        const fp32_4 _x4 = x4[thread_id];
        const fp32_4 _y4 = y4[thread_id];
        fp32_4 tmp;

        if (std::is_same_v<scalar_t, fp32>) {
            tmp.x = _x4.x + _y4.x;
            tmp.y = _x4.y + _y4.y;
            tmp.z = _x4.z + _y4.z;
            tmp.w = _x4.w + _y4.w;
        } else if constexpr (std::is_same_v<scalar_t, c10::Half> || std::is_same_v<scalar_t, c10::BFloat16>) {
            DType<scalar_t> q;

            tmp.x = q.pack(__hadd2(q.unpack(_x4.x), q.unpack(_y4.x)));
            tmp.y = q.pack(__hadd2(q.unpack(_x4.y), q.unpack(_y4.y)));
            tmp.z = q.pack(__hadd2(q.unpack(_x4.z), q.unpack(_y4.z)));
            tmp.w = q.pack(__hadd2(q.unpack(_x4.w), q.unpack(_y4.w)));
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
