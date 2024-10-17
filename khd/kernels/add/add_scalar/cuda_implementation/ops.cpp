#include <torch/extension.h>

void add_scalar_forward_cuda_kernel(
    const torch::Tensor x, const float &y, torch::Tensor output, const int &num_elements, const int &BLOCK_SIZE);

torch::Tensor add_scalar_forward_cuda(torch::Tensor x, const float y, const int BLOCK_SIZE) {
    TORCH_CHECK(x.device().is_cuda(), "tensor x is not on GPU");

    torch::Tensor output = torch::empty_like(x);

    int num_elements = x.numel();

    add_scalar_forward_cuda_kernel(x, y, output, num_elements, BLOCK_SIZE);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_scalar_forward_cuda", &add_scalar_forward_cuda, "Scalar addition forward (CUDA)");
}
