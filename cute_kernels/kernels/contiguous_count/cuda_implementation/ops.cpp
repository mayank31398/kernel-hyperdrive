#include <torch/extension.h>

void swiglu_forward_cuda(const torch::Tensor &gate,
                         const torch::Tensor &up,
                         torch::Tensor &output,
                         const int &vector_instruction_width,
                         const int &BLOCK_SIZE);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swiglu_forward_cuda", &swiglu_forward_cuda, "SwiGLU forward (CUDA)");
}
