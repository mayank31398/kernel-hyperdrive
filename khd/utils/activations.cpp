#include "./dtypes.h"

template <typename T> T sigmoid(const T x) {
    fp32 x_fp32 = DType<T>::upcast(x);

    if (x >= 0) {
        x_fp32 = 1 / (1 + expf(-x_fp32));
    } else {
        x_fp32 = expf(x_fp32);
        x_fp32 = x_fp32 / (1 + x_fp32);
    }

    return DType<T>::downcast(x_fp32);
}

template <typename T, typename T2> __device__ T2 sigmoid2(const T2 x) {
    using dtype = DType<T>;
    T2 output;

    // clang-format off
    #pragma unroll
    // clang-format on
    for (int i = 0; i < 2; i++) {
        T _gate = ((T *)&gate_vec)[i];
        T _up = ((T *)&up_vec)[i];

        output_buffer[i] = dtype::downcast(dtype::upcast(_up) * dtype::upcast(_gate) * sigmoid<T>(_gate));
    }

    return dtype::reinterpret_2x16_as_32_bits(output_buffer[0], output_buffer[1]);
}
