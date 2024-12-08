#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "common.h"

template <>
struct DType<c10::BFloat16> {
    using c10_dtype = c10::BFloat16;
    using nv_dtype = bf16;
    using nv_dtype2 = bf16_2;

    // fp32 -> bf16_2
    inline __device__ static nv_dtype2 reinterpret_32_bits_as_2x16(const fp32 &value) {
        auto [left_int, right_int] = split_fp32_into_16_bits(value);

        nv_dtype left = __ushort_as_bfloat16(left_int);
        nv_dtype right = __ushort_as_bfloat16(right_int);

        return __halves2bfloat162(left, right);
    }

    // bf16_2 -> fp32
    inline __device__ static fp32 reinterpret_2x16_as_32_bits(const nv_dtype2 &value) {
        return reinterpret_2x16_as_32_bits(value.x, value.y);
    }

    // bf16, bf16 -> fp32
    inline __device__ static fp32 reinterpret_2x16_as_32_bits(const nv_dtype &left, const nv_dtype &right) {
        uint16 left_int = __bfloat16_as_ushort(left);
        uint16 right_int = __bfloat16_as_ushort(right);

        return combine_16_bits_into_fp32(left_int, right_int);
    }

    // fp64 -> bf16_2, bf16_2
    inline __device__ static std::tuple<nv_dtype, nv_dtype, nv_dtype, nv_dtype> reinterpret_64_bits_as_4x16(
        const fp64 &value) {
        auto [first_int, second_int, third_int, fourth_int] = split_fp64_into_16_bits(value);

        nv_dtype first = __ushort_as_bfloat16(first_int);
        nv_dtype second = __ushort_as_bfloat16(second_int);
        nv_dtype third = __ushort_as_bfloat16(third_int);
        nv_dtype fourth = __ushort_as_bfloat16(fourth_int);

        return std::make_tuple(first, second, third, fourth);
    }

    // bf16_2, bf16_2 -> fp64
    inline __device__ static fp64 reinterpret_4x16_as_64_bits(const nv_dtype2 &left, const nv_dtype2 &right) {
        uint16 first_int = __bfloat16_as_ushort(left.x);
        uint16 second_int = __bfloat16_as_ushort(left.y);
        uint16 third_int = __bfloat16_as_ushort(right.x);
        uint16 fourth_int = __bfloat16_as_ushort(right.y);

        return combine_16_bits_into_fp64(first_int, second_int, third_int, fourth_int);
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

template <>
struct DType<bf16> : public DType<c10::BFloat16> {};
