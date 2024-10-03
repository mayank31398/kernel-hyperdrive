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

template <> __device__ DType<T>::nv_dtype2 sigmoid<T>(const DType<T>::nv_dtype2 x) {
    using dtype = DType<T>;

    fp32_2 x_fp32 = dtype::upcast(x);
    dtype::nv_dtype2 output_buffer;

    output_buffer.x = sigmoid<T>(x.x);
    output_buffer.y = sigmoid<T>(x.y);

    return dtype::downcast(output_buffer);
}
