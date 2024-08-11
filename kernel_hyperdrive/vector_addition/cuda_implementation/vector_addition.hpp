#include <torch/extension.h>

torch::Tensor vector_addition_forward_kernel_launcher(torch::Tensor x, torch::Tensor y, const int BLOCK_SIZE);
