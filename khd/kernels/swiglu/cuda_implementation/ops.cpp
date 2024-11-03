#include <torch/extension.h>

void swiglu_forward_cuda(torch::Tensor gate, torch::Tensor up, torch::Tensor output, const int BLOCK_SIZE);

void swiglu_backward_cuda_kernel(torch::Tensor gate,
                                 torch::Tensor up,
                                 torch::Tensor output_grad,
                                 torch::Tensor gate_grad,
                                 torch::Tensor up_grad,
                                 const int num_elements,
                                 const int BLOCK_SIZE);

std::vector<torch::Tensor> swiglu_backward_cuda(torch::Tensor gate,
                                                torch::Tensor up,
                                                torch::Tensor output_grad,
                                                const int BLOCK_SIZE) {
    TORCH_CHECK(gate.device().is_cuda(), "tensor gate is not on GPU");
    TORCH_CHECK(up.device().is_cuda(), "tensor up is not on GPU");

    TORCH_CHECK(gate.sizes() == up.sizes(), "tensors gate and up should have same shape");
    TORCH_CHECK(gate.scalar_type() == up.scalar_type(), "tensors gate and up should have same dtype");

    torch::Tensor gate_grad = torch::empty_like(gate);
    torch::Tensor up_grad = torch::empty_like(gate);

    int num_elements = gate.numel();

    swiglu_backward_cuda_kernel(gate, up, output_grad, gate_grad, up_grad, num_elements, BLOCK_SIZE);

    return {gate_grad, up_grad};
}

TORCH_LIBRARY_FRAGMENT(khd, m) {
    m.def("swiglu_forward_cuda(Tensor gate, Tensor up, Tensor output, int num_elements, int BLOCK_SIZE) -> ()");
}

TORCH_LIBRARY_IMPL(khd, CUDA, m) { m.impl("swiglu_forward_cuda", swiglu_forward_cuda); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swiglu_forward_cuda", &swiglu_forward_cuda, "SwiGLU forward (CUDA)");
    m.def("swiglu_backward_cuda", &swiglu_backward_cuda, "SwiGLU backward (CUDA)");
}
