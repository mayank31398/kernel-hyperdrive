#include <torch/extension.h>

void add_scalar_forward_cuda_kernel(
    torch::Tensor x, const float y, torch::Tensor output, const int num_elements, const int BLOCK_SIZE);

torch::Tensor add_scalar_forward_cuda(torch::Tensor x, const float y, const int BLOCK_SIZE) {
    TORCH_CHECK(x.device().is_cuda(), "tensor x is not on GPU");

    if (y == 0) {
        return x;
    }

    torch::Tensor output = torch::empty_like(x);

    int num_elements = x.numel();

    add_scalar_forward_cuda_kernel(x.view(-1), y, output.view(-1), num_elements, BLOCK_SIZE);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_tensor_forward_cuda", &add_tensor_forward_cuda, "Tensor addition forward (CUDA)");
}
