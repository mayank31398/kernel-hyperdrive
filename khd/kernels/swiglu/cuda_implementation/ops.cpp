#include <torch/extension.h>

void swiglu_forward_cuda(torch::Tensor gate,
                         torch::Tensor up,
                         torch::Tensor output,
                         const int &vector_instruction_width,
                         const int BLOCK_SIZE);

void swiglu_backward_cuda(torch::Tensor gate,
                          torch::Tensor up,
                          torch::Tensor output_grad,
                          torch::Tensor gate_grad,
                          torch::Tensor up_grad,
                          const int BLOCK_SIZE);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swiglu_forward_cuda", &swiglu_forward_cuda, "SwiGLU forward (CUDA)");
    m.def("swiglu_backward_cuda", &swiglu_backward_cuda, "SwiGLU backward (CUDA)");
}
