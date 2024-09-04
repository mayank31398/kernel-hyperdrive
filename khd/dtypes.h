#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__device__ std::tuple<uint16_t, uint16_t> get_upper_and_lower_16_bits(float value) {
    uint32_t int_value = __float_as_int(value);

    uint16_t lower_16 = int_value & 0xFFFF;
    uint16_t upper_16 = (int_value >> 16) & 0xFFFF;

    return std::make_tuple(lower_16, upper_16);
}

__device__ float get_float_from_upper_and_lower_16_bits(uint16_t upper_16, uint16_t lower_16) {
    uint32_t int_value = (static_cast<uint32_t>(upper_16) << 16) | lower_16;
    return __int_as_float(int_value);
}

template <typename scalar_t> struct TorchDtype2NVDtype;

template <> struct TorchDtype2NVDtype<c10::Half> {
    using torch_dtype = c10::Half;
    using nv_dtype = half;
    using nv_dtype2 = half2; // vectorized half

    __device__ half2 unpack(float value) {
        auto [lower_16, upper_16] = get_upper_and_lower_16_bits(value);

        half lower_half = __ushort_as_half(lower_16);
        half upper_half = __ushort_as_half(upper_16);

        return __halves2half2(lower_half, upper_half);
    }

    __device__ float pack(half2 value) {
        half lower_half = __low2half(value);
        half upper_half = __high2half(value);

        uint16_t lower_16 = __half_as_short(lower_half);
        uint16_t upper_16 = __half_as_short(upper_half);

        return get_float_from_upper_and_lower_16_bits(upper_16, lower_16);
    }
};

template <> struct TorchDtype2NVDtype<half>;

template <> struct TorchDtype2NVDtype<c10::BFloat16> {
    using torch_dtype = c10::BFloat16;
    using nv_dtype = __nv_bfloat16;
    using nv_dtype2 = __nv_bfloat162; // vectorized bf16

    __device__ __nv_bfloat162 unpack(float value) {
        auto [lower_16, upper_16] = get_upper_and_lower_16_bits(value);

        __nv_bfloat16 lower_half = __ushort_as_bfloat16(lower_16);
        __nv_bfloat16 upper_half = __ushort_as_bfloat16(upper_16);

        return __halves2bfloat162(lower_half, upper_half);
    }

    __device__ float pack(__nv_bfloat162 value) {
        __nv_bfloat16 lower_half = __low2bfloat16(value);
        __nv_bfloat16 upper_half = __high2bfloat16(value);

        uint16_t lower_16 = __bfloat16_as_short(lower_half);
        uint16_t upper_16 = __bfloat16_as_short(upper_half);

        return get_float_from_upper_and_lower_16_bits(upper_16, lower_16);
    }
};

template <> struct TorchDtype2NVDtype<__nv_bfloat16>;
