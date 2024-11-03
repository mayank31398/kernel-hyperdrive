#include <torch/extension.h>

void add_scalar_forward_cuda(const torch::Tensor x, const double &y, torch::Tensor output, const int64_t &BLOCK_SIZE);

TORCH_LIBRARY(khd, m) { m.def("add_scalar_forward_cuda(Tensor x, float y, Tensor output, int BLOCK_SIZE) -> ()"); }

TORCH_LIBRARY_IMPL(khd, CUDA, m) { m.impl("add_scalar_forward_cuda", add_scalar_forward_cuda); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_scalar_forward_cuda", &add_scalar_forward_cuda, "Scalar addition forward (CUDA)");
}
