#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define BLOCK_SIZE 256
#define NUM_ELEMENTS_PER_THREAD 4 // vectorized load store

template <typename scalar_t>
__global__ void vector_addition_forward_kernel(const scalar_t *x,
                                               const scalar_t *y,
                                               scalar_t *output,
                                               const int num_elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (std::is_same_v<scalar_t, float>) {
        if (index * NUM_ELEMENTS_PER_THREAD < num_elements) {
            // float4 is a datatype used for vectorized loads and stores
            float4 *x4 = (float4 *)x;
            float4 *y4 = (float4 *)y;
            float4 *output4 = (float4 *)output;

            // tmp is initialized here to avoid doing multiple writes
            float4 tmp;
            tmp.x = x4[index].x + y4[index].x;
            tmp.y = x4[index].y + y4[index].y;
            tmp.z = x4[index].z + y4[index].z;
            tmp.w = x4[index].w + y4[index].w;

            output4[index] = tmp;
        }
    } else {
        if (index < num_elements) {
            output[index] = x[index] + y[index];
        }
    }
}

torch::Tensor vector_addition_forward_kernel_dispatcher(torch::Tensor x, torch::Tensor y) {
    int num_elements = x.numel();

    torch::Tensor output = torch::empty_like(x);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "vector_addition_forward_kernel", ([&] {
            int num_elements_per_thread = 1;
            if (std::is_same_v<scalar_t, float>) {
                num_elements_per_thread = NUM_ELEMENTS_PER_THREAD;
            }

            int NUM_BLOCKS =
                (num_elements + num_elements_per_thread * BLOCK_SIZE - 1) / (num_elements_per_thread * BLOCK_SIZE);

            vector_addition_forward_kernel<scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                x.data<scalar_t>(), y.data<scalar_t>(), output.data<scalar_t>(), num_elements);
        }));

    return output;
}
