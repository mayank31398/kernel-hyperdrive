#include <torch/extension.h>

void add_tensor_forward_cuda(const torch::Tensor x,
                             const torch::Tensor y,
                             torch::Tensor output,
                             const int64_t &vectorized_loop_size,
                             const int64_t &BLOCK_SIZE);

TORCH_LIBRARY_FRAGMENT(khd, m) {
    m.def(
        "add_tensor_forward_cuda(Tensor x, Tensor y, Tensor output, int vectorized_loop_size, int BLOCK_SIZE) -> ()");
}

TORCH_LIBRARY_IMPL(khd, CUDA, m) { m.impl("add_tensor_forward_cuda", add_tensor_forward_cuda); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_tensor_forward_cuda", &add_tensor_forward_cuda, "Tensor addition forward (CUDA)");
}
