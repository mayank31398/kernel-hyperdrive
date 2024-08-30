#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void vector_addition_forward_kernel(const scalar_t *x,
                                               const scalar_t *y,
                                               scalar_t *output,
                                               const int total_elements,
                                               const int num_elements_per_thread) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

#pragma unroll
    for (int i = 0; i < n; i++) {
        int index = num_elements_per_thread * i + total_elements;
        if (index < num_elements) {
            output[index] = x[index] + y[index];
        }
    }
}

void vector_addition_forward_kernel_dispatcher(torch::Tensor x,
                                               torch::Tensor y,
                                               torch::Tensor output,
                                               const int num_elements_per_thread,
                                               const int NUM_BLOCKS,
                                               const int BLOCK_SIZE) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "vector_addition_forward_kernel", ([&] {
            vector_addition_forward_kernel<scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                x.data<scalar_t>(), y.data<scalar_t>(), output.data<scalar_t>(), x.numel(), num_elements_per_thread);
        }));
}
