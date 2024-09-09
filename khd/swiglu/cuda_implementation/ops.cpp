#include <torch/extension.h>

torch::Tensor swiglu_forward_kernel_dispatcher(torch::Tensor x, torch::Tensor y);

torch::Tensor swiglu_forward(torch::Tensor gate, torch::Tensor up) {
    TORCH_CHECK(gate.device().is_cuda(), "tensor gate is not on GPU")
    TORCH_CHECK(up.device().is_cuda(), "tensor up is not on GPU")

    gate.shape up.view()

        TORCH_CHECK(x.numel() == y.numel(), "both tensors should have same number of elements");
    TORCH_CHECK(x.scalar_type() == y.scalar_type(), "both tensors should have same dtype");

    return vector_addition_forward_kernel_dispatcher(x, y);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_addition_forward", &vector_addition_forward, "Vector addition forward (CUDA)");
}
