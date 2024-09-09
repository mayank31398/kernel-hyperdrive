#include <torch/extension.h>

void vector_addition_forward_cuda_kernel(
    torch::Tensor x, torch::Tensor y, torch::Tensor output, const int num_elements, const int BLOCK_SIZE);

torch::Tensor vector_addition_forward(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.device().is_cuda(), "tensor x is not on GPU");
    TORCH_CHECK(y.device().is_cuda(), "tensor y is not on GPU");

    TORCH_CHECK(x.sizes() == y.sizes(), "tensor x and y should be of the same sizes");
    TORCH_CHECK(x.scalar_type() == y.scalar_type(), "both tensors should have same dtype");

    int num_elements = x.numel();
    const int BLOCK_SIZE = 1024;

    torch::Tensor output = torch::empty_like(x);

    vector_addition_forward_cuda_kernel(x, y, output, num_elements, BLOCK_SIZE);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_addition_forward", &vector_addition_forward, "Vector addition forward (CUDA)");
}
