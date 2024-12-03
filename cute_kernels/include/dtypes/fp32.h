#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "common.h"
#include "fp64.h"

template <>
struct DType<fp32> : public DType<fp64> {
    using c10_dtype = fp32;
    using nv_dtype = fp32;
    using nv_dtype2 = fp32_2;
    using nv_dtype4 = fp32_4;

    inline __device__ static nv_dtype2 reinterpret_64_bits_as_2x32(const fp64 &value) {
        auto [left_int, right_int] = split_fp64_into_32_bits(value);

        nv_dtype left = __uint_as_float(left_int);
        nv_dtype right = __uint_as_float(right_int);

        return make_float2(left, right);
    }

    inline __device__ static fp64 reinterpret_2x32_as_64_bits(const nv_dtype &lower_half, const nv_dtype &upper_half) {
        uint32 lower_32 = __float_as_uint(lower_half);
        uint32 upper_32 = __float_as_uint(upper_half);

        return get_fp64_from_upper_and_lower_32_bits(upper_32, lower_32);
    }

    inline __device__ static fp32 reinterpret_2x32_as_64_bits(const nv_dtype2 &value) {
        return reinterpret_2x32_as_64_bits(value.x, value.y);
    }

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
