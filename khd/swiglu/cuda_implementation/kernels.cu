#include "../../utils/activations.cpp"
#include "../../utils/dtypes.h"
#include "../../utils/threads.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void _swiglu_forward_cuda_kernel(const scalar_t *gate,
                                            const scalar_t *up,
                                            scalar_t *output,
                                            const int num_elements) {
    const int thread_id = get_global_thread_id();
    const int num_elements_per_thread = get_num_elements_in_vector_dtype<scalar_t, fp32_4>();

    const int start = thread_id * num_elements_per_thread;
    const int end = (thread_id + 1) * num_elements_per_thread - 1; // inclusive of last element

    if (start < num_elements && end < num_elements) {
        const fp32 *gate_vec = (fp32 *)&((const fp32_4 *)gate)[thread_id];
        const fp32 *up_vec = (fp32 *)&((const fp32_4 *)up)[thread_id];

        fp32 output_buffer[4];

        // clang-format off
        #pragma unroll
        // clang-format on
        for (int i = 0; i < 4; i++) {
            if (std::is_same_v<scalar_t, fp32>) {
                output_buffer[i] = up_vec[i] * gate_vec[i] * sigmoid<scalar_t>(gate_vec[i]);
            } else if constexpr (std::is_same_v<scalar_t, c10::Half> || std::is_same_v<scalar_t, c10::BFloat16>) {
                output_buffer[i] = _swiglu_forward_vectorized_16(gate_vec[i], up_vec[i]);
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

torch::Tensor swiglu_forward_cuda_kernel(
    torch::Tensor gate, torch::Tensor up, torch::Tensor output, const int num_elements, const int BLOCK_SIZE) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, gate.scalar_type(), "vector_addition_forward_kernel", ([&] {
            const int num_elements_per_thread = get_num_elements_in_vector_dtype<scalar_t, fp32_4>();

            const int num_elements_per_block = BLOCK_SIZE * num_elements_per_thread;
            const int NUM_BLOCKS = (num_elements + num_elements_per_block - 1) / num_elements_per_block;

            _swiglu_forward_cuda_kernel<scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                gate.data_ptr<scalar_t>(), up.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), num_elements);
        }));

    return output;
}