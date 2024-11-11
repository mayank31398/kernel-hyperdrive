#include <torch/extension.h>

void embedding_forward_cuda(const torch::Tensor x,
                            const torch::Tensor weight,
                            torch::Tensor output,
                            const int &BLOCK_SIZE);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("embedding_forward_cuda", &embedding_forward_cuda, "Embedding forward (CUDA)");
}
