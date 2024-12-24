#include <torch/extension.h>

void contigous_count_cuda(const torch::Tensor &x,
                          const torch::Tensor &output,
                          const int C,
                          const int &BLOCK_SIZE_B,
                          const int &BLOCK_SIZE_C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("contigous_count_cuda", &contigous_count_cuda, "contiguous count forward (CUDA)");
}
