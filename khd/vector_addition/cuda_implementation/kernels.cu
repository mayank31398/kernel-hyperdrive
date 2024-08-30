#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void vector_addition_forward_kernel(const scalar_t *x,
                                               const scalar_t *y,
                                               scalar_t *output,
                                               const int num_elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_elements) {
        output[index] = x[index] + y[index];
    }
}

void vector_addition_forward_kernel_dispatcher(
    torch::Tensor x, torch::Tensor y, torch::Tensor output, const int NUM_BLOCKS, const int BLOCK_SIZE) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "vector_addition_forward_kernel", ([&] {
            vector_addition_forward_kernel<scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                x.data<scalar_t>(), y.data<scalar_t>(), output.data<scalar_t>(), x.numel());
        }));
}
