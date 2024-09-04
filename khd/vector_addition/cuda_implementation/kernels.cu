#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define BLOCK_SIZE 1024

// for vectorized load store
#define NUM_ELEMENTS_PER_THREAD_FP32 4
#define NUM_ELEMENTS_PER_THREAD_FP16 8
#define NUM_ELEMENTS_PER_THREAD_BF16 8

__device__ std::tuple<uint16_t, uint16_t> get_upper_and_lower_16_bits(float value) {
    uint32_t int_value = __float_as_int(value);

    uint16_t lower_16 = int_value & 0xFFFF;
    uint16_t upper_16 = (int_value >> 16) & 0xFFFF;

    return std::make_tuple(lower_16, upper_16);
}

template <typename scalar_t, typename scalar_t2> __device__ scalar_t2 unpack(float value);

template <> __device__ half2 unpack<half, half2>(float value) {
    auto [lower_16, upper_16] = get_upper_and_lower_16_bits(value);

    half lower_half = __ushort_as_half(lower_16);
    half upper_half = __ushort_as_half(upper_16);

    return __halves2half2(lower_half, upper_half);
}

template <> __device__ __nv_bfloat162 unpack<__nv_bfloat16, __nv_bfloat162>(float value) {
    auto [lower_16, upper_16] = get_upper_and_lower_16_bits(value);

    __nv_bfloat16 lower_half = __ushort_as_bfloat16(lower_16);
    __nv_bfloat16 upper_half = __ushort_as_bfloat16(upper_16);

    return __halves2bfloat162(lower_half, upper_half);
}

__device__ float get_float_from_upper_and_lower_16_bits(uint16_t upper_16, uint16_t lower_16) {
    uint32_t int_value = (static_cast<uint32_t>(upper_16) << 16) | lower_16;
    return __int_as_float(int_value);
}

template <typename scalar_t2> __device__ float pack(scalar_t2 value);

template <> __device__ float pack<half2>(half2 value) {
    half lower_half = __low2half(value);
    half upper_half = __high2half(value);

    uint16_t lower_16 = __half_as_short(lower_half);
    uint16_t upper_16 = __half_as_short(upper_half);

    return get_float_from_upper_and_lower_16_bits(upper_16, lower_16);
}

template <> __device__ float pack<__nv_bfloat162>(__nv_bfloat162 value) {
    __nv_bfloat16 lower_half = __low2bfloat16(value);
    __nv_bfloat16 upper_half = __high2bfloat16(value);

    uint16_t lower_16 = __bfloat16_as_short(lower_half);
    uint16_t upper_16 = __bfloat16_as_short(upper_half);

    return get_float_from_upper_and_lower_16_bits(upper_16, lower_16);
}

template <typename scalar_t>
__global__ void vector_addition_forward_kernel(const scalar_t *x,
                                               const scalar_t *y,
                                               scalar_t *output,
                                               const int num_elements,
                                               const int num_elements_per_thread) {
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    const int start = thread_id * num_elements_per_thread;
    const int end = (thread_id + 1) * num_elements_per_thread - 1; // inclusive of last element

    if (start < num_elements && end < num_elements) {
        // float4 is a datatype used for vectorized loads and stores
        const float4 *x4 = (const float4 *)x;
        const float4 *y4 = (const float4 *)y;
        float4 *output4 = (float4 *)output;

        // tmp is initialized here to avoid doing multiple writes
        const float4 _x4 = x4[thread_id];
        const float4 _y4 = y4[thread_id];
        float4 tmp;

        if (std::is_same_v<scalar_t, float>) {
            tmp.x = _x4.x + _y4.x;
            tmp.y = _x4.y + _y4.y;
            tmp.z = _x4.z + _y4.z;
            tmp.w = _x4.w + _y4.w;
        } else if (std::is_same_v<scalar_t, c10::Half>) {
            tmp.x = pack<half2>(__hadd2(unpack<half, half2>(_x4.x), unpack<half, half2>(_y4.x)));
            tmp.y = pack<half2>(__hadd2(unpack<half, half2>(_x4.y), unpack<half, half2>(_y4.y)));
            tmp.z = pack<half2>(__hadd2(unpack<half, half2>(_x4.z), unpack<half, half2>(_y4.z)));
            tmp.w = pack<half2>(__hadd2(unpack<half, half2>(_x4.w), unpack<half, half2>(_y4.w)));
        } else if (std::is_same_v<scalar_t, c10::BFloat16>) {
            tmp.x = pack<__nv_bfloat162>(
                __hadd2(unpack<__nv_bfloat16, __nv_bfloat162>(_x4.x), unpack<__nv_bfloat16, __nv_bfloat162>(_y4.x)));
            tmp.y = pack<__nv_bfloat162>(
                __hadd2(unpack<__nv_bfloat16, __nv_bfloat162>(_x4.y), unpack<__nv_bfloat16, __nv_bfloat162>(_y4.y)));
            tmp.z = pack<__nv_bfloat162>(
                __hadd2(unpack<__nv_bfloat16, __nv_bfloat162>(_x4.z), unpack<__nv_bfloat16, __nv_bfloat162>(_y4.z)));
            tmp.w = pack<__nv_bfloat162>(
                __hadd2(unpack<__nv_bfloat16, __nv_bfloat162>(_x4.w), unpack<__nv_bfloat16, __nv_bfloat162>(_y4.w)));
        }

        output4[thread_id] = tmp;
    } else if (start < num_elements) {
#pragma unroll
        for (int i = start; i < num_elements; i++) {
            output[i] = x[i] + y[i];
        }
    }
}

torch::Tensor vector_addition_forward_kernel_dispatcher(torch::Tensor x, torch::Tensor y) {
    int num_elements = x.numel();

    torch::Tensor output = torch::empty_like(x);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "vector_addition_forward_kernel", ([&] {
            int num_elements_per_thread;
            if (std::is_same_v<scalar_t, float>) {
                num_elements_per_thread = NUM_ELEMENTS_PER_THREAD_FP32;
            } else if (std::is_same_v<scalar_t, c10::Half>) {
                num_elements_per_thread = NUM_ELEMENTS_PER_THREAD_FP16;
            } else if (std::is_same_v<scalar_t, c10::BFloat16>) {
                num_elements_per_thread = NUM_ELEMENTS_PER_THREAD_BF16;
            }

            int num_elements_per_block = BLOCK_SIZE * num_elements_per_thread;
            int NUM_BLOCKS = (num_elements + num_elements_per_block - 1) / num_elements_per_block;

            vector_addition_forward_kernel<scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(x.data<scalar_t>(),
                                                                                 y.data<scalar_t>(),
                                                                                 output.data<scalar_t>(),
                                                                                 num_elements,
                                                                                 num_elements_per_thread);
        }));

    return output;
}
