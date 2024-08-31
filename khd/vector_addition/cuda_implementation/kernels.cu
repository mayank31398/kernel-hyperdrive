#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define BLOCK_SIZE 128
#define NUM_ELEMENTS_PER_THREAD 4

template <typename scalar_t>
__global__ void vector_addition_forward_kernel(const scalar_t *x,
                                               const scalar_t *y,
                                               scalar_t *output,
                                               const int num_elements) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = gridDim.x * BLOCK_SIZE;

#pragma unroll
    for (int i = 0; i < NUM_ELEMENTS_PER_THREAD; i++) {
        int index = num_threads * i + thread_id;
        if (index < num_elements) {
            output[index] = x[index] + y[index];
        }
    }
}

torch::Tensor vector_addition_forward_kernel_dispatcher(torch::Tensor x, torch::Tensor y) {
    int num_elements = x.numel();

    int num_elements_per_block = BLOCK_SIZE * NUM_ELEMENTS_PER_THREAD;
    int NUM_BLOCKS = (num_elements + num_elements_per_block - 1) / num_elements_per_block;

    torch::Tensor output = torch::empty_like(x);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "vector_addition_forward_kernel", ([&] {
            vector_addition_forward_kernel<scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                x.data<scalar_t>(), y.data<scalar_t>(), output.data<scalar_t>(), num_elements);
        }));

    return output;
}
