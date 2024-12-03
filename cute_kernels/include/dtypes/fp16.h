#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "common.h"

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

template <>
struct DType<fp16> : public DType<c10::Half> {};
