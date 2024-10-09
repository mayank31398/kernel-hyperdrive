#include <torch/extension.h>

void add_tensor_forward_cuda_kernel(const torch::Tensor x,
                                    const torch::Tensor y,
                                    torch::Tensor output,
                                    const bool &use_efficient_kernel,
                                    const int &num_elements,
                                    const int &BLOCK_SIZE);

torch::Tensor add_tensor_forward_cuda(const torch::Tensor x,
                                      const torch::Tensor y,
                                      const bool &use_efficient_kernel,
                                      const int &BLOCK_SIZE) {
    TORCH_CHECK(x.device().is_cuda(), "tensor x is not on GPU");
    TORCH_CHECK(y.device().is_cuda(), "tensor y is not on GPU");

    TORCH_CHECK(x.sizes() == y.sizes(), "tensors x and y should have same shape");
    TORCH_CHECK(x.scalar_type() == y.scalar_type(), "tensors x and y should have same dtype");

    torch::Tensor output = torch::empty_like(x);

    int num_elements = x.numel();

    add_tensor_forward_cuda_kernel(
        x.view(-1), y.view(-1), output.view(-1), use_efficient_kernel, num_elements, BLOCK_SIZE);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_tensor_forward_cuda", &add_tensor_forward_cuda, "Tensor addition forward (CUDA)");
}
