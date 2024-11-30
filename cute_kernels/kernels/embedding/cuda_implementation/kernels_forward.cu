#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../../utils/activations.h"
#include "../../utils/dtypes.h"
#include "../../utils/threads.h"

template <typename scalar_t, typename vector_t>
__global__ void _embedding_forward_cuda_kernel(const scalar_t *input_ids,
                                               const scalar_t *weight,
                                               scalar_t *output,
                                               const int64_t num_elements,
                                               const int &BLOCK_SIZE_B,
                                               const int &BLOCK_SIZE_H) {
    constexpr int vector_instruction_width = sizeof(vector_t) / sizeof(scalar_t);
    const int64_t thread_id = get_global_thread_id();

    using dtype = DType<scalar_t>;
}

void embedding_forward_cuda(const torch::Tensor &input_ids,
                            const torch::Tensor &weight,
                            torch::Tensor output,
                            const int &vector_instruction_width,
                            const int &BLOCK_SIZE_B,
                            const int &BLOCK_SIZE_H) {
    const int64_t num_elements = gate.numel();

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(
        gate.scalar_type(), "embedding_forward_cuda_kernel", ([&] {
            const int num_elements_per_block = BLOCK_SIZE * vector_instruction_width;
            const int NUM_BLOCKS = (num_elements + num_elements_per_block - 1) / num_elements_per_block;

            switch (vector_instruction_width) {
                case 1:
                    _swiglu_forward_cuda_kernel<scalar_t, scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                        gate.data_ptr<scalar_t>(), up.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), num_elements);
                    break;
                // case 2:
                //     using vector_t = typename DType<scalar_t>::nv_dtype2;
                //     _swiglu_forward_cuda_kernel<scalar_t, vector_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                //         gate.data_ptr<scalar_t>(), up.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                //         num_elements);
                //     break;
                // case 4:
                //     if constexpr (std::is_same_v<scalar_t, fp32>) {
                //         _swiglu_forward_cuda_kernel<scalar_t, fp32_4>
                //             <<<NUM_BLOCKS, BLOCK_SIZE>>>(gate.data_ptr<scalar_t>(),
                //                                          up.data_ptr<scalar_t>(),
                //                                          output.data_ptr<scalar_t>(),
                //                                          num_elements);
                //     } else {
                //         _swiglu_forward_cuda_kernel<scalar_t, fp32_2>
                //             <<<NUM_BLOCKS, BLOCK_SIZE>>>(gate.data_ptr<scalar_t>(),
                //                                          up.data_ptr<scalar_t>(),
                //                                          output.data_ptr<scalar_t>(),
                //                                          num_elements);
                //     }
                //     break;
                // case 8:
                //     if constexpr (std::is_same_v<scalar_t, fp32>) {
                //         throw std::runtime_error("fp32 doesn't support vector_instruction_width = 8");
                //     } else {
                //         _swiglu_forward_cuda_kernel<scalar_t, fp32_4>
                //             <<<NUM_BLOCKS, BLOCK_SIZE>>>(gate.data_ptr<scalar_t>(),
                //                                          up.data_ptr<scalar_t>(),
                //                                          output.data_ptr<scalar_t>(),
                //                                          num_elements);
                //     }
                //     break;
                default:
                    throw std::runtime_error("invalid vector_instruction_width");
                    break;
            }
        }));
}
