#include <torch/extension.h>

void swiglu_forward_cuda(torch::Tensor gate, torch::Tensor up, torch::Tensor output, const int64_t BLOCK_SIZE);

void swiglu_backward_cuda(torch::Tensor gate,
                          torch::Tensor up,
                          torch::Tensor output_grad,
                          torch::Tensor gate_grad,
                          torch::Tensor up_grad,
                          const int64_t BLOCK_SIZE);

TORCH_LIBRARY_FRAGMENT(khd, m) {
    m.def("swiglu_forward_cuda(Tensor gate, Tensor up, Tensor output, int BLOCK_SIZE) -> ()");
    m.def("swiglu_backward_cuda(Tensor gate, Tensor up, Tensor output_grad, Tensor gate_grad, Tensor up_grad, int "
          "BLOCK_SIZE) -> ()");
}

TORCH_LIBRARY_IMPL(khd, CUDA, m) {
    m.impl("swiglu_forward_cuda", swiglu_forward_cuda);
    m.impl("swiglu_backward_cuda", swiglu_backward_cuda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swiglu_forward_cuda", &swiglu_forward_cuda, "SwiGLU forward (CUDA)");
    m.def("swiglu_backward_cuda", &swiglu_backward_cuda, "SwiGLU backward (CUDA)");
}
