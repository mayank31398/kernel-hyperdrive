#include <torch/extension.h>

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " is not on CUDA device")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " is not a contiguous tensor")

#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x);
