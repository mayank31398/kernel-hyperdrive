#include <torch/extension.h>
#include "../../utils.h"

// CUDA kernel declarations
torch::Tensor vector_addition_forward_kernel_launcher(torch::Tensor x, torch::Tensor y, const int BLOCK_SIZE);

torch::Tensor vector_addition_forward(torch::Tensor x, torch::Tensor y)
{
    CHECK_INPUT(x);
    CHECK_INPUT(y);

    TORCH_CHECK(x.dim() == 1, "tensor should be 1 dimensional")
    TORCH_CHECK(y.dim() == 1, "tensor should be 1 dimensional")

    TORCH_CHECK(x.numel() == y.numel(), "both tensors should have same number of elements");
    TORCH_CHECK(x.type() == y.type(), "both tensors should have same dtype");

    return vector_addition_forward_kernel_launcher(x, y, 1024);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("vector_addition_forward", &vector_addition_forward, "Vector addition forward (CUDA)");
}
