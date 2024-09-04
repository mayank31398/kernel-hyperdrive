#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define BLOCK_SIZE 1024

// for vectorized load store
#define NUM_ELEMENTS_PER_THREAD_FP32 4
#define NUM_ELEMENTS_PER_THREAD_FP16 8
#define NUM_ELEMENTS_PER_THREAD_BF16 8

__device__ half2 unpack(float value) {
    uint32_t intValue = __float_as_int(value);

    uint16_t lower16Bits = intValue & 0xFFFF;
    uint16_t higher16Bits = (intValue >> 16) & 0xFFFF;

    half lowerHalf = __ushort_as_half(lower16Bits);
    half higherHalf = __ushort_as_half(higher16Bits);

    return __halves2half2(lowerHalf, higherHalf);
}

__device__ float pack(half2 value) {
    // Extract the two half values from half2
    half lowerHalf = __low2half(value);
    half higherHalf = __high2half(value);

    // Convert the half values to uint16_t
    uint16_t lower16Bits = __half_as_short(lowerHalf);
    uint16_t higher16Bits = __half_as_short(higherHalf);

    // Combine the 16-bit values into a single 32-bit integer
    uint32_t intValue = (static_cast<uint32_t>(higher16Bits) << 16) | lower16Bits;

    // Convert the 32-bit integer back to float
    return __int_as_float(intValue);
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
            tmp.x = pack(unpack(_x4.x) + unpack(_y4.x));
            tmp.y = pack(unpack(_x4.y) + unpack(_y4.y));
            tmp.z = pack(unpack(_x4.z) + unpack(_y4.z));
            tmp.w = pack(unpack(_x4.w) + unpack(_y4.w));
        } else if (std::is_same_v<scalar_t, c10::BFloat16>) {
            tmp.x = pack(unpack(_x4.x) + unpack(_y4.x));
            tmp.y = pack(unpack(_x4.y) + unpack(_y4.y));
            tmp.z = pack(unpack(_x4.z) + unpack(_y4.z));
            tmp.w = pack(unpack(_x4.w) + unpack(_y4.w));
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
