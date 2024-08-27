#include <torch/extension.h>

// CUDA kernel declarations
torch::Tensor vector_addition_forward_kernel_launcher(torch::Tensor x, torch::Tensor y, const int BLOCK_SIZE);

torch::Tensor vector_addition_forward(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.device().is_cuda(), "tensor x is not on GPU")
    TORCH_CHECK(y.device().is_cuda(), "tensor y is not on GPU")

    TORCH_CHECK(x.is_contiguous(), "tensor x is not a contiguous")
    TORCH_CHECK(y.is_contiguous(), "tensor y is not a contiguous")

    TORCH_CHECK(x.dim() == 1, "tensor x should be 1 dimensional")
    TORCH_CHECK(y.dim() == 1, "tensor y should be 1 dimensional")

    int num_elements = x.numel();
    TORCH_CHECK(num_elements == y.numel(), "both tensors should have same number of elements");
    TORCH_CHECK(x.type() == y.type(), "both tensors should have same dtype");

    // TODO use num_elements

    return vector_addition_forward_kernel_launcher(x, y, BLOCK_SIZE);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_addition_forward", &vector_addition_forward, "Vector addition forward (CUDA)");
}
