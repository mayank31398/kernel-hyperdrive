#include "../../utils/activations.cpp"
#include "../../utils/dtypes.h"
#include "../../utils/threads.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// for vectorized load store
std::unordered_map<std::type_index, int> num_elements_per_thread_mapping = {
    {typeid(fp32), 4}, {typeid(c10::Half), 8}, {typeid(c10::BFloat16), 8}};

template <typename scalar_t>
__global__ void _swiglu_forward_cuda_kernel(const scalar_t *gate,
                                            const scalar_t *up,
                                            scalar_t *output,
                                            const int num_elements,
                                            const int num_elements_per_thread) {
    const int thread_id = get_global_thread_id();

    const int start = thread_id * num_elements_per_thread;
    const int end = (thread_id + 1) * num_elements_per_thread - 1; // inclusive of last element

    if (start < num_elements && end < num_elements) {
        // fp32_4 is a datatype used for vectorized loads and stores
        const fp32_4 *gate4 = (const fp32_4 *)gate;
        const fp32_4 *up4 = (const fp32_4 *)up;
        fp32_4 *output4 = (fp32_4 *)output;

        const fp32 *_gate = (fp32 *)(&gate4[thread_id]);
        const fp32 *_up = (fp32 *)(&up4[thread_id]);

        // tmp is initialized here to avoid doing multiple writes
        fp32_4 tmp4;
        fp32 *tmp = (fp32 *)(&tmp4);

        // clang-format off
        #pragma unroll
        // clang-format on
        for (int i = 0; i < 4; i++) {
            if (std::is_same_v<scalar_t, fp32>) {
                tmp[i] = up * gate * sigmoid(gate);
            } else if constexpr (std::is_same_v<scalar_t, c10::Half> || std::is_same_v<scalar_t, c10::BFloat16>) {
                DType<scalar_t> q;
                tmp[i] = q.pack_to_fp32(__hadd2(q.unpack_from_fp32(_x[i]), q.unpack_from_fp32(_y[i])));
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

torch::Tensor swiglu_forward_cuda_kernel(
    torch::Tensor gate, torch::Tensor up, torch::Tensor output, const int num_elements, const int BLOCK_SIZE) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "vector_addition_forward_kernel", ([&] {
            int num_elements_per_thread = num_elements_per_thread_mapping[std::type_index(typeid(scalar_t))];

            int num_elements_per_block = BLOCK_SIZE * num_elements_per_thread;
            int NUM_BLOCKS = (num_elements + num_elements_per_block - 1) / num_elements_per_block;

            vector_addition_forward_kernel<scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(gate.data_ptr<scalar_t>(),
                                                                                 up.data_ptr<scalar_t>(),
                                                                                 output.data_ptr<scalar_t>(),
                                                                                 num_elements,
                                                                                 num_elements_per_thread);
        }));

    return output;
}
