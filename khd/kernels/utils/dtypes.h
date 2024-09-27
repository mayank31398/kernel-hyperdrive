#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// define dtype aliases
#define fp32 float
#define fp32_2 float2
#define fp32_4 float4

#define fp16 half
#define fp16_2 half2

#define bf16 __nv_bfloat16
#define bf16_2 __nv_bfloat162

__device__ std::tuple<uint16_t, uint16_t> get_upper_and_lower_16_bits_from_fp32(fp32 value) {
    uint32_t int_value = __float_as_int(value);

    uint16_t lower_16 = int_value & 0xFFFF;
    uint16_t upper_16 = (int_value >> 16) & 0xFFFF;

    return std::make_tuple(lower_16, upper_16);
}

__device__ fp32 get_fp32_from_upper_and_lower_16_bits(uint16_t upper_16, uint16_t lower_16) {
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

    __device__ static nv_dtype upcast(nv_dtype value) { return value; }
    __device__ static nv_dtype2 upcast(nv_dtype2 value) { return value; }
    __device__ static nv_dtype4 upcast(nv_dtype4 value) { return value; }

    __device__ static nv_dtype downcast(nv_dtype value) { return value; }
    __device__ static nv_dtype2 downcast(nv_dtype2 value) { return value; }
    __device__ static nv_dtype4 downcast(nv_dtype4 value) { return value; }
};

// struct for c10::Half
template <> struct DType<c10::Half> {
    using c10_dtype = c10::Half;
    using nv_dtype = fp16;
    using nv_dtype2 = fp16_2;

    __device__ static nv_dtype2 reinterpret_32_bits_as_2x16(fp32 value) {
        auto [lower_16, upper_16] = get_upper_and_lower_16_bits_from_fp32(value);

        nv_dtype lower_half = __ushort_as_half(lower_16);
        nv_dtype upper_half = __ushort_as_half(upper_16);

        return __halves2half2(lower_half, upper_half);
    }

    __device__ static fp32 reinterpret_2x16_as_32_bits(nv_dtype lower_half, nv_dtype upper_half) {
        uint16_t lower_16 = __half_as_short(lower_half);
        uint16_t upper_16 = __half_as_short(upper_half);

        return get_fp32_from_upper_and_lower_16_bits(upper_16, lower_16);
    }

    __device__ static fp32 reinterpret_2x16_as_32_bits(nv_dtype2 value) {
        nv_dtype lower_half = __low2half(value);
        nv_dtype upper_half = __high2half(value);

        return reinterpret_2x16_as_32_bits(lower_half, upper_half);
    }

    __device__ static fp32 upcast(const nv_dtype value) { return __half2float(value); }
    __device__ static fp32_2 upcast(const nv_dtype2 value) { return __half22float2(value); }

    __device__ static nv_dtype downcast(const fp32 value) { return __float2half(value); }
    __device__ static nv_dtype2 downcast(const fp32_2 value) { return __float22half2_rn(value); }

    __device__ static nv_dtype2 make2(const nv_dtype value) { return __half2half2(value); }
};

// struct for half (basically another alias for the above)
template <> struct DType<fp16> : public DType<c10::Half> {};

// struct for c10::BFloat16
template <> struct DType<c10::BFloat16> {
    using c10_dtype = c10::BFloat16;
    using nv_dtype = bf16;
    using nv_dtype2 = bf16_2;

    __device__ static nv_dtype2 reinterpret_32_bits_as_2x16(fp32 value) {
        auto [lower_16, upper_16] = get_upper_and_lower_16_bits_from_fp32(value);

        nv_dtype lower_half = __ushort_as_bfloat16(lower_16);
        nv_dtype upper_half = __ushort_as_bfloat16(upper_16);

        return __halves2bfloat162(lower_half, upper_half);
    }

    __device__ static fp32 reinterpret_2x16_as_32_bits(nv_dtype lower_half, nv_dtype upper_half) {
        uint16_t lower_16 = __bfloat16_as_short(lower_half);
        uint16_t upper_16 = __bfloat16_as_short(upper_half);

        return get_fp32_from_upper_and_lower_16_bits(upper_16, lower_16);
    }

    __device__ static fp32 reinterpret_2x16_as_32_bits(nv_dtype2 value) {
        nv_dtype lower_half = __low2bfloat16(value);
        nv_dtype upper_half = __high2bfloat16(value);

        return reinterpret_2x16_as_32_bits(lower_half, upper_half);
    }

    __device__ static fp32 upcast(const nv_dtype value) { return __bfloat162float(value); }
    __device__ static fp32_2 upcast(const nv_dtype2 value) { return __bfloat1622float2(value); }

    __device__ static nv_dtype downcast(const fp32 value) { return __float2bfloat16(value); }
    __device__ static nv_dtype2 downcast(const fp32_2 value) { return __float22bfloat162_rn(value); }

    __device__ static nv_dtype2 make2(const nv_dtype value) { return __bfloat162bfloat162(value); }
};

// struct for bf16 (basically another alias for the above)
template <> struct DType<bf16> : public DType<c10::BFloat16> {};
