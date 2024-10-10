#include <torch/extension.h>

torch::Tensor add_tensor_forward_cuda_kernel_dispatch(const torch::Tensor x,
                                                      const torch::Tensor y,
                                                      const int &vectorized_load_store_size,
                                                      const int &BLOCK_SIZE);

torch::Tensor add_tensor_forward_cuda(const torch::Tensor x,
                                      const torch::Tensor y,
                                      const int &vectorized_load_store_size,
                                      const int &BLOCK_SIZE) {
    TORCH_CHECK(x.device().is_cuda(), "tensor x is not on GPU");
    TORCH_CHECK(y.device().is_cuda(), "tensor y is not on GPU");

    TORCH_CHECK(x.sizes() == y.sizes(), "tensors x and y should have same shape");
    TORCH_CHECK(x.scalar_type() == y.scalar_type(), "tensors x and y should have same dtype");

    torch::Tensor output = add_tensor_forward_cuda_kernel_dispatch(x, y, vectorized_load_store_size, BLOCK_SIZE);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_tensor_forward_cuda", &add_tensor_forward_cuda, "Tensor addition forward (CUDA)");
}
