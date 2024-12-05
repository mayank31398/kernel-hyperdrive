#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../../../../include/dtypes/all.h"
#include "../../../../include/threads.h"

template <typename scalar_t>
__global__ void _add_scalar_forward_cuda_kernel_fp32_fp16_bf16_1(const scalar_t *x,
                                                                 const fp32 y,
                                                                 scalar_t *output,
                                                                 const int64_t num_elements);

void add_scalar_forward_cuda(const torch::Tensor &x,
                             const float &y,
                             torch::Tensor &output,
                             const int &vector_instruction_width,
                             const int &BLOCK_SIZE) {
    assert(BLOCK_SIZE % WARP_SIZE == 0);

    const int64_t num_elements = x.numel();

    const int num_elements_per_block = BLOCK_SIZE * vector_instruction_width;
    const int NUM_BLOCKS = (num_elements + num_elements_per_block - 1) / num_elements_per_block;

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(
        x.scalar_type(), "add_scalar_forward_cuda_kernel", ([&] {
            switch (vector_instruction_width) {
                case 1:
                    _add_scalar_forward_cuda_kernel_fp32_fp16_bf16_1<scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                        x.data_ptr<scalar_t>(), y, output.data_ptr<scalar_t>(), num_elements);
                    break;
                // case 2:
                //     using vector_t = typename DType<scalar_t>::nv_dtype2;
                //     _add_scalar_forward_cuda_kernel<scalar_t, vector_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                //         x.data_ptr<scalar_t>(), y, output.data_ptr<scalar_t>(), num_elements);
                //     break;
                // case 4:
                //     if constexpr (std::is_same_v<scalar_t, fp32>) {
                //         _add_scalar_forward_cuda_kernel<scalar_t, fp32_4><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                //             x.data_ptr<scalar_t>(), y, output.data_ptr<scalar_t>(), num_elements);
                //     } else {
                //         _add_scalar_forward_cuda_kernel<scalar_t, fp32_2><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                //             x.data_ptr<scalar_t>(), y, output.data_ptr<scalar_t>(), num_elements);
                //     }
                //     break;
                // case 8:
                //     if constexpr (std::is_same_v<scalar_t, fp32>) {
                //         _add_scalar_forward_cuda_kernel<scalar_t, fp64_4><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                //             x.data_ptr<scalar_t>(), y, output.data_ptr<scalar_t>(), num_elements);
                //     } else {
                //         _add_scalar_forward_cuda_kernel<scalar_t, fp32_4><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                //             x.data_ptr<scalar_t>(), y, output.data_ptr<scalar_t>(), num_elements);
                //     }
                //     break;
                // case 16:
                //     if constexpr (std::is_same_v<scalar_t, c10::Half> || std::is_same_v<scalar_t, c10::BFloat16>) {
                //         _add_scalar_forward_cuda_kernel<scalar_t, fp64_4><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                //             x.data_ptr<scalar_t>(), y, output.data_ptr<scalar_t>(), num_elements);
                //     } else {
                //         throw std::runtime_error("invalid vector_instruction_width = 16 for fp32");
                //     }
                //     break;
                default:
                    throw std::runtime_error("invalid vector_instruction_width");
                    break;
            }
        }));
}
