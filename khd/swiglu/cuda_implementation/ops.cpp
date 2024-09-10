#include <torch/extension.h>

void swiglu_forward_cuda_kernel(
    torch::Tensor gate, torch::Tensor up, torch::Tensor output, const int num_elements, const int BLOCK_SIZE);

void swiglu_backward_cuda_kernel(torch::Tensor gate,
                                 torch::Tensor up,
                                 torch::Tensor output_grad,
                                 torch::Tensor gate_grad,
                                 torch::Tensor up_grad,
                                 const int num_elements,
                                 const int BLOCK_SIZE);

torch::Tensor swiglu_forward_cuda(torch::Tensor gate,
                                  torch::Tensor up,
                                  const bool memory_efficient,
                                  const int BLOCK_SIZE) {
    TORCH_CHECK(gate.device().is_cuda(), "tensor gate is not on GPU");
    TORCH_CHECK(up.device().is_cuda(), "tensor up is not on GPU");

    TORCH_CHECK(gate.sizes() == up.sizes(), "tensors gate and up should have same shape");
    TORCH_CHECK(gate.scalar_type() == up.scalar_type(), "tensors gate and up should have same dtype");

    if (memory_efficient && (gate.is_leaf() || up.is_leaf())) {
        throw runtime_error("leaf variables can't be used in an in-place operation");
    }

    torch::Tensor output = torch::empty_like(gate);

    int num_elements = gate.numel();

    swiglu_forward_cuda_kernel(gate.view(-1), up.view(-1), output.view(-1), num_elements, BLOCK_SIZE);

    return output;
}

torch::Tensor swiglu_backward_cuda(torch::Tensor gate,
                                   torch::Tensor up,
                                   torch::Tensor output_grad,
                                   const bool memory_efficient,
                                   const int BLOCK_SIZE) {
    int num_elements = gate.numel();

    torch::Tensor gate_grad, up_grad;
    if (memory_efficient) {
        gate_grad = gate;
        up_grad = up;
    } else {
        gate_grad = torch::empty_like(gate);
        up_grad = torch::empty_like(up);
    }

    swiglu_backward_cuda_kernel(gate.view(-1), up.view(-1), output.view(-1), num_elements, BLOCK_SIZE);

    return {gate_grad, up_grad};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swiglu_forward_cuda", &swiglu_forward_cuda, "SwiGLU forward (CUDA)");
    m.def("swiglu_backward_cuda", &swiglu_backward_cuda, "SwiGLU backward (CUDA)");
}
