#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define AT_DISPATCH_CASE_CUSTOM_FLOAT_TYPES(...)            \
    AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)     \
    AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__) \
    AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)

#define AT_DISPATCH_CUSTOM_FLOAT_TYPES(TYPE, NAME, ...) \
    AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_CUSTOM_FLOAT_TYPES(__VA_ARGS__))

// define dtype aliases
using fp32 = float;
using fp32_2 = float2;
using fp32_4 = float4;

using fp16 = half;
using fp16_2 = half2;

using bf16 = __nv_bfloat16;
using bf16_2 = __nv_bfloat162;

using int16 = short;
using uint16 = ushort;

using int32 = int;
using uint32 = uint;

inline __device__ std::tuple<uint16, uint16> get_upper_and_lower_16_bits_from_fp32(const fp32 &value) {
    uint32 int_value = __float_as_uint(value);

    uint16 lower_16 = int_value & 0xFFFF;
    uint16 upper_16 = int_value >> 16;

    return std::make_tuple(lower_16, upper_16);
}

inline __device__ fp32 get_fp32_from_upper_and_lower_16_bits(const uint16 &upper_16, const uint16 &lower_16) {
    uint32 int_value = (static_cast<uint32>(upper_16) << 16) | lower_16;
    return __uint_as_float(int_value);
}

// base struct for converting torch ScalarType to NVIDIA's dtype
template <typename scalar_t>
struct DType {
    using c10_dtype = scalar_t;
};
