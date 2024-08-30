#include <torch/extension.h>

torch::Tensor vector_addition_forward_kernel_dispatcher(torch::Tensor x,
                                                        torch::Tensor y,
                                                        const int NUM_BLOCKS,
                                                        const int BLOCK_SIZE);

torch::Tensor vector_addition_forward(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.device().is_cuda(), "tensor x is not on GPU")
    TORCH_CHECK(y.device().is_cuda(), "tensor y is not on GPU")

    TORCH_CHECK(x.is_contiguous(), "tensor x is not a contiguous")
    TORCH_CHECK(y.is_contiguous(), "tensor y is not a contiguous")

    TORCH_CHECK(x.dim() == 1, "tensor x should be 1 dimensional")
    TORCH_CHECK(y.dim() == 1, "tensor y should be 1 dimensional")

    int num_elements = x.numel();

    TORCH_CHECK(y.numel() == num_elements, "both tensors should have same number of elements");
    TORCH_CHECK(x.scalar_type() == y.scalar_type(), "both tensors should have same dtype");

    int BLOCK_SIZE = 1024;
    int NUM_BLOCKS = (int)ceil((float)num_elements / BLOCK_SIZE);

    return vector_addition_forward_kernel_dispatcher(x, y, NUM_BLOCKS, BLOCK_SIZE);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_addition_forward", &vector_addition_forward, "Vector addition forward (CUDA)");
}
