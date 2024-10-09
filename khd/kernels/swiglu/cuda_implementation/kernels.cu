#include "../../utils/activations.h"
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

    using dtype = DType<scalar_t>;
    using T = typename dtype::nv_dtype;
    using T2 = typename dtype::nv_dtype2;

    if (start < num_elements && end < num_elements) {
        const fp32 *gate_vec = (fp32 *)&((const fp32_4 *)gate)[thread_id];
        const fp32 *up_vec = (fp32 *)&((const fp32_4 *)up)[thread_id];

        fp32 output_buffer[4];

        // clang-format off
        #pragma unroll
        // clang-format on
        for (int i = 0; i < 4; i++) {
            if (std::is_same_v<scalar_t, fp32>) {
                output_buffer[i] = up_vec[i] * gate_vec[i] * sigmoid<fp32, fp32>(gate_vec[i]);
            } else if constexpr (std::is_same_v<scalar_t, c10::Half> || std::is_same_v<scalar_t, c10::BFloat16>) {
                fp32_2 _up = dtype::upcast(dtype::reinterpret_32_bits_as_2x16(up_vec[i]));
                fp32_2 _gate = dtype::upcast(dtype::reinterpret_32_bits_as_2x16(gate_vec[i]));

                _gate = DType<fp32>::make2(_up.x * _gate.x * sigmoid<fp32, fp32>(_gate.x),
                                           _up.y * _gate.y * sigmoid<fp32, fp32>(_gate.y));
                output_buffer[i] = dtype::reinterpret_2x16_as_32_bits(dtype::downcast(_gate));
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
            fp32 _gate = dtype::upcast(static_cast<T>(gate[i]));
            output[i] = dtype::downcast(dtype::upcast(static_cast<T>(up[i])) * _gate * sigmoid<fp32, fp32>(_gate));
        }
    }
}

void swiglu_forward_cuda_kernel(
    torch::Tensor gate, torch::Tensor up, torch::Tensor output, const int num_elements, const int BLOCK_SIZE) {
    AT_DISPATCH_CUSTOM_FLOAT_TYPES(
        gate.scalar_type(), "swiglu_forward_cuda_kernel", ([&] {
            const int num_elements_per_thread = get_num_elements_in_vector_dtype<scalar_t, fp32_4>();

            const int num_elements_per_block = BLOCK_SIZE * num_elements_per_thread;
            const int NUM_BLOCKS = (num_elements + num_elements_per_block - 1) / num_elements_per_block;

            _swiglu_forward_cuda_kernel<scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                gate.data_ptr<scalar_t>(), up.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), num_elements);
        }));
}
