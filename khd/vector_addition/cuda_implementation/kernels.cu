#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define BLOCK_SIZE 1024

#define NUM_ELEMENTS_PER_THREAD_FP32 4 // vectorized load store
#define NUM_ELEMENTS_PER_THREAD_FP16 2 // vectorized load store
#define NUM_ELEMENTS_PER_THREAD_BF16 2 // vectorized load store

template <typename scalar_t>
__global__ void vector_addition_forward_kernel(const scalar_t *x,
                                               const scalar_t *y,
                                               scalar_t *output,
                                               const int num_elements,
                                               const int num_elements_per_thread) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index * num_elements_per_thread < num_elements) {
        if (std::is_same_v<scalar_t, float>) {
            // float4 is a datatype used for vectorized loads and stores
            float4 *x4 = (float4 *)x;
            float4 *y4 = (float4 *)y;
            float4 *output4 = (float4 *)output;

            // tmp is initialized here to avoid doing multiple writes
            float4 _x4 = x4[index];
            float4 _y4 = y4[index];
            float4 tmp;

            tmp.x = _x4.x + _y4.x;
            tmp.y = _x4.y + _y4.y;
            tmp.z = _x4.z + _y4.z;
            tmp.w = _x4.w + _y4.w;

            output4[index] = tmp;
        } else if (std::is_same_v<scalar_t, c10::Half>) {
            __half2 *x2 = (__half2 *)x;
            __half2 *y2 = (__half2 *)y;
            __half2 *output2 = (__half2 *)output;

            output2[index] = __hadd2(x2[index], y2[index]);
        } else {
            __nv_bfloat162 *x2 = (__nv_bfloat162 *)x;
            __nv_bfloat162 *y2 = (__nv_bfloat162 *)y;
            __nv_bfloat162 *output2 = (__nv_bfloat162 *)output;

            output2[index] = __hadd2(x2[index], y2[index]);
        }
    }
}

torch::Tensor vector_addition_forward_kernel_dispatcher(torch::Tensor x, torch::Tensor y) {
    int num_elements = x.numel();

    torch::Tensor output = torch::empty_like(x);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "vector_addition_forward_kernel", ([&] {
            int num_elements_per_thread;
            if (std::is_same_v<scalar_t, float>) {
                num_elements_per_thread = NUM_ELEMENTS_PER_THREAD_FP32;
            } else if (std::is_same_v<scalar_t, c10::Half>) {
                num_elements_per_thread = NUM_ELEMENTS_PER_THREAD_FP16;
            } else if (std::is_same_v<scalar_t, c10::BFloat16>) {
                num_elements_per_thread = NUM_ELEMENTS_PER_THREAD_BF16;
            }

            int NUM_BLOCKS =
                (num_elements + num_elements_per_thread * BLOCK_SIZE - 1) / (num_elements_per_thread * BLOCK_SIZE);

            vector_addition_forward_kernel<scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(x.data<scalar_t>(),
                                                                                 y.data<scalar_t>(),
                                                                                 output.data<scalar_t>(),
                                                                                 num_elements,
                                                                                 num_elements_per_thread);
        }));

    return output;
}
