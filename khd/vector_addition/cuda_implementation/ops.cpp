#include <torch/extension.h>

void vector_addition_forward_cuda(
    torch::Tensor x, torch::Tensor y, torch::Tensor output, const int num_elements, const int BLOCK_SIZE);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_addition_forward_cuda", &vector_addition_forward_cuda, "Vector addition forward (CUDA)");
}
