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
                tmp[i] = _up[i] * _gate[i] * sigmoid(_gate[i]);
            } else if constexpr (std::is_same_v<scalar_t, c10::Half> || std::is_same_v<scalar_t, c10::BFloat16>) {
                using dtype = DType<scalar_t>;
                using T = typename dtype::nv_dtype;
                using T2 = typename dtype::nv_dtype2;

                T2 gate1 = dtype::reinterpret_32_bits_as_2x16(_gate[i]);
                T2 up1 = dtype::reinterpret_32_bits_as_2x16(_up[i]);
                T2 tmp1;

                // clang-format off
                #pragma unroll
                // clang-format on
                for (int j = 0; j < 2; j++) {
                    T *_gate1 = (T *)(&gate1);
                    T *_up1 = (T *)(&up1);

                    fp32 _gate1_fp32 = dtype::upcast(_gate1[j]);
                    fp32 _up1_fp32 = dtype::upcast(_up1[j]);

                    fp32 _tmp_fp32 = _up1_fp32 * _gate1_fp32 * sigmoid(_gate1_fp32);
                    tmp1[j] = dtype::downcast(_tmp_fp32);
                }

                tmp[i] = dtype::reinterpret_2x16_as_32_bits(tmp1);
            } else {
                assert(false && "Function not implemented");
            }
        }

        output4[thread_id] = tmp4;
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
