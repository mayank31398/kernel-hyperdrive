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
// float
using fp16 = half;
using fp16_2 = half2;

using bf16 = __nv_bfloat16;
using bf16_2 = __nv_bfloat162;

using fp32 = float;
using fp32_2 = float2;
using fp32_4 = float4;

using fp64 = double;
using fp64_2 = double2;
using fp64_4 = double4;

// int
using int16 = short;
using uint16 = ushort;

using int32 = int;
using uint32 = uint;

using int64 = int64_t;
using uint64 = uint64_t;

inline __device__ std::tuple<uint32, uint32> get_upper_and_lower_32_bits_from_fp64(const fp64 &value) {
    uint64 int_value = __double_as_longlong(value);

    uint32 lower_16 = int_value & 0xFFFF;
    uint32 upper_16 = int_value >> 16;

    return std::make_tuple(lower_16, upper_16);
}

inline __device__ fp64 get_fp64_from_upper_and_lower_32_bits(const uint32 &upper_16, const uint32 &lower_16) {
    uint64 int_value = (static_cast<uint64>(upper_16) << 16) | lower_16;
    return __longlong_as_double(static_cast<int64>(int_value));
}

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

// struct for c10::Half
template <>
struct DType<fp64> {
    using c10_dtype = fp64;
    using nv_dtype = fp64;
    using nv_dtype2 = fp64_2;
    using nv_dtype4 = fp64_4;

    inline __device__ static nv_dtype upcast(const nv_dtype &value) { return value; }
    inline __device__ static nv_dtype2 upcast(const nv_dtype2 &value) { return value; }
    inline __device__ static nv_dtype4 upcast(const nv_dtype4 &value) { return value; }

    inline __device__ static nv_dtype downcast(const nv_dtype &value) { return value; }
    inline __device__ static nv_dtype2 downcast(const nv_dtype2 &value) { return value; }
    inline __device__ static nv_dtype4 downcast(const nv_dtype4 &value) { return value; }

    inline __device__ static nv_dtype2 make2(const nv_dtype &x, const nv_dtype &y) { return make_double2(x, y); }
    inline __device__ static nv_dtype2 make2(const nv_dtype *array) { return make_double2(array[0], array[1]); }

    inline __device__ static nv_dtype4 make4(const nv_dtype &x,
                                             const nv_dtype &y,
                                             const nv_dtype &z,
                                             const nv_dtype &t) {
        return make_double4(x, y, z, t);
    }
    inline __device__ static nv_dtype4 make4(const nv_dtype *array) {
        return make_double4(array[0], array[1], array[2], array[3]);
    }
};

// struct for c10::Float
template <>
struct DType<fp32> {
    using c10_dtype = fp32;
    using nv_dtype = fp32;
    using nv_dtype2 = fp32_2;
    using nv_dtype4 = fp32_4;

    inline __device__ static nv_dtype2 reinterpret_64_bits_as_2x32(const fp64 &value) {
        auto [lower_16, upper_16] = get_upper_and_lower_32_bits_from_fp64(value);

        nv_dtype lower_half = __uint_as_float(lower_16);
        nv_dtype upper_half = __uint_as_float(upper_16);

        return make_float2(lower_half, upper_half);
    }

    inline __device__ static fp64 reinterpret_2x32_as_64_bits(const nv_dtype &lower_half, const nv_dtype &upper_half) {
        uint32 lower_16 = __float_as_uint(lower_half);
        uint32 upper_16 = __float_as_uint(upper_half);

        return get_fp64_from_upper_and_lower_32_bits(upper_16, lower_16);
    }

    inline __device__ static fp32 reinterpret_2x32_as_64_bits(const nv_dtype2 &value) {
        return reinterpret_2x32_as_64_bits(value.x, value.y);
    }

    inline __device__ static fp32 upcast(const nv_dtype &value) { return value; }
    inline __device__ static fp32_2 upcast(const nv_dtype2 &value) { return value; }
    inline __device__ static fp32_4 upcast(const nv_dtype4 &value) { return value; }

    inline __device__ static nv_dtype downcast(const nv_dtype &value) { return value; }
    inline __device__ static nv_dtype2 downcast(const nv_dtype2 &value) { return value; }
    inline __device__ static nv_dtype4 downcast(const nv_dtype4 &value) { return value; }

    inline __device__ static nv_dtype2 make2(const nv_dtype &x, const nv_dtype &y) { return make_float2(x, y); }
    inline __device__ static nv_dtype2 make2(const nv_dtype *array) { return make_float2(array[0], array[1]); }

    inline __device__ static nv_dtype4 make4(const nv_dtype &x,
                                             const nv_dtype &y,
                                             const nv_dtype &z,
                                             const nv_dtype &t) {
        return make_float4(x, y, z, t);
    }
    inline __device__ static nv_dtype4 make4(const nv_dtype *array) {
        return make_float4(array[0], array[1], array[2], array[3]);
    }
};

// struct for c10::Half
template <>
struct DType<c10::Half> {
    using c10_dtype = c10::Half;
    using nv_dtype = fp16;
    using nv_dtype2 = fp16_2;

    inline __device__ static nv_dtype2 reinterpret_32_bits_as_2x16(const fp32 &value) {
        auto [lower_16, upper_16] = get_upper_and_lower_16_bits_from_fp32(value);

        nv_dtype lower_half = __ushort_as_half(lower_16);
        nv_dtype upper_half = __ushort_as_half(upper_16);

        return __halves2half2(lower_half, upper_half);
    }

    inline __device__ static fp32 reinterpret_2x16_as_32_bits(const nv_dtype &lower_half, const nv_dtype &upper_half) {
        uint16 lower_16 = __half_as_ushort(lower_half);
        uint16 upper_16 = __half_as_ushort(upper_half);

        return get_fp32_from_upper_and_lower_16_bits(upper_16, lower_16);
    }

    inline __device__ static fp32 reinterpret_2x16_as_32_bits(const nv_dtype2 &value) {
        nv_dtype lower_half = __low2half(value);
        nv_dtype upper_half = __high2half(value);

        return reinterpret_2x16_as_32_bits(lower_half, upper_half);
    }

    inline __device__ static fp32 upcast(const c10_dtype &value) { return upcast(static_cast<nv_dtype>(value)); }
    inline __device__ static fp32 upcast(const nv_dtype &value) { return __half2float(value); }
    inline __device__ static fp32_2 upcast(const nv_dtype2 &value) { return __half22float2(value); }

    inline __device__ static nv_dtype downcast(const fp32 &value) { return __float2half(value); }
    inline __device__ static nv_dtype2 downcast(const fp32_2 &value) { return __float22half2_rn(value); }

    inline __device__ static nv_dtype2 make2(const nv_dtype &value) { return __half2half2(value); }
    inline __device__ static nv_dtype2 make2(const nv_dtype &x, const nv_dtype &y) { return make_half2(x, y); }
    inline __device__ static nv_dtype2 make2(const nv_dtype *array) { return make_half2(array[0], array[1]); }
};

// struct for half (basically another alias for the above)
template <>
struct DType<fp16> : public DType<c10::Half> {};

// struct for c10::BFloat16
template <>
struct DType<c10::BFloat16> {
    using c10_dtype = c10::BFloat16;
    using nv_dtype = bf16;
    using nv_dtype2 = bf16_2;

    inline __device__ static nv_dtype2 reinterpret_32_bits_as_2x16(const fp32 &value) {
        auto [lower_16, upper_16] = get_upper_and_lower_16_bits_from_fp32(value);

        nv_dtype lower_half = __ushort_as_bfloat16(lower_16);
        nv_dtype upper_half = __ushort_as_bfloat16(upper_16);

        return __halves2bfloat162(lower_half, upper_half);
    }

    inline __device__ static fp32 reinterpret_2x16_as_32_bits(const nv_dtype &lower_half, const nv_dtype &upper_half) {
        uint16 lower_16 = __bfloat16_as_ushort(lower_half);
        uint16 upper_16 = __bfloat16_as_ushort(upper_half);

        return get_fp32_from_upper_and_lower_16_bits(upper_16, lower_16);
    }

    inline __device__ static fp32 reinterpret_2x16_as_32_bits(const nv_dtype2 &value) {
        nv_dtype lower_half = __low2bfloat16(value);
        nv_dtype upper_half = __high2bfloat16(value);

        return reinterpret_2x16_as_32_bits(lower_half, upper_half);
    }

    inline __device__ static fp32 upcast(const c10_dtype &value) { return upcast(static_cast<nv_dtype>(value)); }
    inline __device__ static fp32 upcast(const nv_dtype &value) { return __bfloat162float(value); }
    inline __device__ static fp32_2 upcast(const nv_dtype2 &value) { return __bfloat1622float2(value); }

    inline __device__ static nv_dtype downcast(const fp32 &value) { return __float2bfloat16(value); }
    inline __device__ static nv_dtype2 downcast(const fp32_2 &value) { return __float22bfloat162_rn(value); }

    inline __device__ static nv_dtype2 make2(const nv_dtype &value) { return __bfloat162bfloat162(value); }
    inline __device__ static nv_dtype2 make2(const nv_dtype &x, const nv_dtype &y) { return make_bfloat162(x, y); }
    inline __device__ static nv_dtype2 make2(const nv_dtype *array) { return make_bfloat162(array[0], array[1]); }
};

// struct for bf16 (basically another alias for the above)
template <>
struct DType<bf16> : public DType<c10::BFloat16> {};
