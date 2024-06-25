#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void vector_addition_forward_kernel(const scalar_t* x,
                                               const scalar_t* y,
                                               scalar_t* output,
                                               const int num_elements)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_elements) output[index] = x[index] + y[index];
}

torch::Tensor vector_addition_forward_kernel_launcher(torch::Tensor x, torch::Tensor y, const int BLOCK_SIZE)
{
    int num_elements = x.numel();
    torch::Tensor output = torch::empty_like(x);

    int blocks = (int)ceil((float)num_elements / BLOCK_SIZE);

    if (at::isReducedFloatingType(x.scalar_type())) {
        AT_DISPATCH_REDUCED_FLOATING_TYPES(
            x.scalar_type(), "vector_addition_forward_kernel", ([&] {
                vector_addition_forward_kernel<scalar_t><<<blocks, BLOCK_SIZE>>>(
                    x.data<scalar_t>(), y.data<scalar_t>(), output.data<scalar_t>(), num_elements);
            }));
    } else {
        AT_DISPATCH_FLOATING_TYPES(
            x.scalar_type(), "vector_addition_forward_kernel", ([&] {
                vector_addition_forward_kernel<scalar_t><<<blocks, BLOCK_SIZE>>>(
                    x.data<scalar_t>(), y.data<scalar_t>(), output.data<scalar_t>(), num_elements);
            }));
    }

    return output;
}
