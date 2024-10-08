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

#define HALF_MASK 0xFFFF

__device__ std::tuple<uint16_t, uint16_t> get_upper_and_lower_16_bits_from_fp32(const fp32 &value) {
    uint32_t int_value = __float_as_int(value);

    uint16_t lower_16 = int_value & HALF_MASK;
    uint16_t upper_16 = (int_value >> 16) & HALF_MASK;

    return std::make_tuple(lower_16, upper_16);
}

__device__ fp32 get_fp32_from_upper_and_lower_16_bits(const uint16_t &upper_16, const uint16_t &lower_16) {
    uint32_t int_value = (static_cast<uint32_t>(upper_16) << 16) | lower_16;
    return __int_as_float(int_value);
}

template <typename T, typename vecT> __host__ __device__ int get_num_elements_in_vector_dtype() {
    return sizeof(vecT) / sizeof(T);
}

// base struct for converting torch ScalarType to NVIDIA's dtype
template <typename scalar_t> struct DType {
    using c10_dtype = scalar_t;
};

// struct for c10::Float
template <> struct DType<fp32> {
    using c10_dtype = fp32;
    using nv_dtype = fp32;
    using nv_dtype2 = fp32_2;
    using nv_dtype4 = fp32_4;

    __device__ static fp32 upcast(const nv_dtype &value) { return value; }
    __device__ static fp32_2 upcast(const nv_dtype2 &value) { return value; }
    __device__ static fp32_4 upcast(const nv_dtype4 &value) { return value; }

    __device__ static nv_dtype downcast(const nv_dtype &value) { return value; }
    __device__ static nv_dtype2 downcast(const nv_dtype2 &value) { return value; }
    __device__ static nv_dtype4 downcast(const nv_dtype4 &value) { return value; }

    __device__ static nv_dtype2 make2(const nv_dtype &x, const nv_dtype &y) { return make_float2(x, y); }
    __device__ static nv_dtype2 make2(const nv_dtype *array) { return make_float2(array[0], array[1]); }

    __device__ static nv_dtype4 make4(const nv_dtype &x, const nv_dtype &y, const nv_dtype &z, const nv_dtype &t) {
        return make_float4(x, y, z, t);
    }
    __device__ static nv_dtype4 make4(const nv_dtype *array) {
        return make_float4(array[0], array[1], array[2], array[3]);
    }
};

// struct for c10::Half
template <> struct DType<c10::Half> {
    using c10_dtype = c10::Half;
    using nv_dtype = fp16;
    using nv_dtype2 = fp16_2;

    __device__ static nv_dtype2 reinterpret_32_bits_as_2x16(const fp32 &value) {
        auto [lower_16, upper_16] = get_upper_and_lower_16_bits_from_fp32(value);

        nv_dtype lower_half = __ushort_as_half(lower_16);
        nv_dtype upper_half = __ushort_as_half(upper_16);

        return __halves2half2(lower_half, upper_half);
    }

    __device__ static fp32 reinterpret_2x16_as_32_bits(const nv_dtype &lower_half, const nv_dtype &upper_half) {
        uint16_t lower_16 = __half_as_short(lower_half);
        uint16_t upper_16 = __half_as_short(upper_half);

        return get_fp32_from_upper_and_lower_16_bits(upper_16, lower_16);
    }

    __device__ static fp32 reinterpret_2x16_as_32_bits(const nv_dtype2 &value) {
        nv_dtype lower_half = __low2half(value);
        nv_dtype upper_half = __high2half(value);

        return reinterpret_2x16_as_32_bits(lower_half, upper_half);
    }

    __device__ static fp32 upcast(const c10_dtype &value) { return upcast(static_cast<nv_dtype>(value)); }
    __device__ static fp32 upcast(const nv_dtype &value) { return __half2float(value); }
    __device__ static fp32_2 upcast(const nv_dtype2 &value) { return __half22float2(value); }

    __device__ static nv_dtype downcast(const fp32 &value) { return __float2half(value); }
    __device__ static nv_dtype2 downcast(const fp32_2 &value) { return __float22half2_rn(value); }

    __device__ static nv_dtype2 make2(const nv_dtype &value) { return __half2half2(value); }
    __device__ static nv_dtype2 make2(const nv_dtype &x, const nv_dtype &y) { return make_half2(x, y); }
    __device__ static nv_dtype2 make2(const nv_dtype *array) { return make_half2(array[0], array[1]); }
};

// struct for half (basically another alias for the above)
template <> struct DType<fp16> : public DType<c10::Half> {};

// struct for c10::BFloat16
template <> struct DType<c10::BFloat16> {
    using c10_dtype = c10::BFloat16;
    using nv_dtype = bf16;
    using nv_dtype2 = bf16_2;

    __device__ static nv_dtype2 reinterpret_32_bits_as_2x16(const fp32 &value) {
        auto [lower_16, upper_16] = get_upper_and_lower_16_bits_from_fp32(value);

        nv_dtype lower_half = __ushort_as_bfloat16(lower_16);
        nv_dtype upper_half = __ushort_as_bfloat16(upper_16);

        return __halves2bfloat162(lower_half, upper_half);
    }

    __device__ static fp32 reinterpret_2x16_as_32_bits(const nv_dtype &lower_half, const nv_dtype &upper_half) {
        uint16_t lower_16 = __bfloat16_as_short(lower_half);
        uint16_t upper_16 = __bfloat16_as_short(upper_half);

        return get_fp32_from_upper_and_lower_16_bits(upper_16, lower_16);
    }

    __device__ static fp32 reinterpret_2x16_as_32_bits(nv_dtype2 value) {
        nv_dtype lower_half = __low2bfloat16(value);
        nv_dtype upper_half = __high2bfloat16(value);

        return reinterpret_2x16_as_32_bits(lower_half, upper_half);
    }

    __device__ static fp32 upcast(const c10_dtype &value) { return upcast(static_cast<nv_dtype>(value)); }
    __device__ static fp32 upcast(const nv_dtype &value) { return __bfloat162float(value); }
    __device__ static fp32_2 upcast(const nv_dtype2 &value) { return __bfloat1622float2(value); }

    __device__ static nv_dtype downcast(const fp32 &value) { return __float2bfloat16(value); }
    __device__ static nv_dtype2 downcast(const fp32_2 &value) { return __float22bfloat162_rn(value); }

    __device__ static nv_dtype2 make2(const nv_dtype &value) { return __bfloat162bfloat162(value); }
    __device__ static nv_dtype2 make2(const nv_dtype &x, const nv_dtype &y) { return make_bfloat162(x, y); }
    __device__ static nv_dtype2 make2(const nv_dtype *array) { return make_bfloat162(array[0], array[1]); }
};

// struct for bf16 (basically another alias for the above)
template <> struct DType<bf16> : public DType<c10::BFloat16> {};
