#pragma once

#include "./dtypes.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <int vectorized_load_store_size, typename scalar_t> struct VectorDTypeSelector;

// Specialization for vectorized_load_store_size == 1
template <typename scalar_t> struct VectorDTypeSelector<1, scalar_t> {
    using vector_t = scalar_t;
};

// Specialization for vectorized_load_store_size == 2
template <typename scalar_t> struct VectorDTypeSelector<2, scalar_t> {
    using vector_t = std::conditional_t<
        std::is_same_v<scalar_t, fp32>,
        fp32_2,
        std::conditional_t<std::is_same_v<scalar_t, c10::Half> || std::is_same_v<scalar_t, c10::BFloat16>,
                           typename DType<scalar_t>::nv_dtype2,
                           void // Fallback for invalid cases
                           >>;
};

// Specialization for vectorized_load_store_size == 4
template <typename scalar_t> struct VectorDTypeSelector<4, scalar_t> {
    using vector_t = std::conditional_t<
        std::is_same_v<scalar_t, fp32>,
        fp32_4,
        std::conditional_t<std::is_same_v<scalar_t, c10::Half> || std::is_same_v<scalar_t, c10::BFloat16>,
                           fp32_2,
                           void // Fallback for invalid cases
                           >>;
};

// Specialization for vectorized_load_store_size == 8
template <typename scalar_t> struct VectorDTypeSelector<8, scalar_t> {
    using vector_t = std::conditional_t<
        std::is_same_v<scalar_t, fp32>,
        void, // Invalid case
        std::conditional_t<std::is_same_v<scalar_t, c10::Half> || std::is_same_v<scalar_t, c10::BFloat16>,
                           fp32_4,
                           void // Fallback for invalid cases
                           >>;
};

static_assert(std::is_same_v<VectorDTypeSelector<8, fp32>, void>, "Invalid vectorized_load_store_size");
