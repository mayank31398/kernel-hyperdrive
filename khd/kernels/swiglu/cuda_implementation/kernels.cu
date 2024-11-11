#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../../utils/activations.h"
#include "../../utils/dtypes.h"
#include "../../utils/threads.h"

template <typename scalar_t>
__global__ void _swiglu_forward_cuda_kernel(const scalar_t *gate,
                                            const scalar_t *up,
                                            scalar_t *output,
                                            const int64_t num_elements) {
    const int64_t thread_id = get_global_thread_id();
    const int vector_instruction_width = sizeof(fp32_4) / sizeof(scalar_t);

    const int64_t start = thread_id * vector_instruction_width;
    const int64_t end = (thread_id + 1) * vector_instruction_width - 1;  // inclusive of last element

    using dtype = DType<scalar_t>;
    using T = typename dtype::nv_dtype;

    if (start < num_elements && end < num_elements) {
        const fp32 *_gate = (fp32 *)&((fp32_4 *)gate)[thread_id];
        const fp32 *_up = (fp32 *)&((fp32_4 *)up)[thread_id];

        fp32 output_buffer[4];

        // clang-format off
        #pragma unroll
        // clang-format on
        for (int i = 0; i < 4; i++) {
            if constexpr (std::is_same_v<scalar_t, fp32>) {
                output_buffer[i] = _up[i] * _gate[i] * sigmoid<fp32, fp32>(_gate[i]);
            } else {
                fp32_2 _up_upcast = dtype::upcast(dtype::reinterpret_32_bits_as_2x16(_up[i]));
                fp32_2 _gate_upcast = dtype::upcast(dtype::reinterpret_32_bits_as_2x16(_gate[i]));

                _gate_upcast = DType<fp32>::make2(_up_upcast.x * _gate_upcast.x * sigmoid<fp32, fp32>(_gate_upcast.x),
                                                  _up_upcast.y * _gate_upcast.y * sigmoid<fp32, fp32>(_gate_upcast.y));
                output_buffer[i] = dtype::reinterpret_2x16_as_32_bits(dtype::downcast(_gate_upcast));
            }
        }

        ((fp32_4 *)output)[thread_id] = DType<fp32>::make4(output_buffer);
    } else if (start < num_elements) {
        // clang-format off
        #pragma unroll
        // clang-format on
        for (int64_t i = start; i < num_elements; i++) {
            fp32 _gate = dtype::upcast(static_cast<T>(gate[i]));
            output[i] = dtype::downcast(dtype::upcast(static_cast<T>(up[i])) * _gate * sigmoid<fp32, fp32>(_gate));
        }
    }
}

template <typename scalar_t>
__global__ void _swiglu_backward_cuda_kernel(const scalar_t *gate,
                                             const scalar_t *up,
                                             const scalar_t *output_grad,
                                             scalar_t *gate_grad,
                                             scalar_t *up_grad,
                                             const int64_t num_elements) {
    const int64_t thread_id = get_global_thread_id();
    const int vector_instruction_width = sizeof(fp32_4) / sizeof(scalar_t);

    const int64_t start = thread_id * vector_instruction_width;
    const int64_t end = (thread_id + 1) * vector_instruction_width - 1;  // inclusive of last element

    using dtype = DType<scalar_t>;
    using T = typename dtype::nv_dtype;
    using T2 = typename dtype::nv_dtype2;

    if (start < num_elements && end < num_elements) {
        const fp32 *_gate = (fp32 *)&((const fp32_4 *)gate)[thread_id];
        const fp32 *_up = (fp32 *)&((const fp32_4 *)up)[thread_id];
        const fp32 *_output_grad = (fp32 *)&((const fp32_4 *)output_grad)[thread_id];

        fp32 gate_grad_buffer[4];
        fp32 up_grad_buffer[4];

        // clang-format off
        #pragma unroll
        // clang-format on
        for (int i = 0; i < 4; i++) {
            if constexpr (std::is_same_v<scalar_t, fp32>) {
                fp32 gate_sigmoid = sigmoid<fp32, fp32>(_gate[i]);
                fp32 gate_silu = _gate[i] * gate_sigmoid;

                up_grad_buffer[i] = _output_grad[i] * gate_silu;
                gate_grad_buffer[i] = _output_grad[i] * _up[i] * (gate_sigmoid + gate_silu * (1 - gate_sigmoid));
            } else {
                fp32_2 _up_upcast = dtype::upcast(dtype::reinterpret_32_bits_as_2x16(_up[i]));
                fp32_2 _gate_upcast = dtype::upcast(dtype::reinterpret_32_bits_as_2x16(_gate[i]));
                fp32_2 _output_grad_upcast = dtype::upcast(dtype::reinterpret_32_bits_as_2x16(_output_grad[i]));

                fp32_2 _gate_grad;
                fp32_2 _up_grad;

                fp32 gate_sigmoid = sigmoid<fp32, fp32>(_gate_upcast.x);
                fp32 gate_silu = _gate_upcast.x * gate_sigmoid;
                _up_grad.x = _output_grad_upcast.x * gate_silu;
                _gate_grad.x = _output_grad_upcast.x * _up_upcast.x * (gate_sigmoid + gate_silu * (1 - gate_sigmoid));

                gate_sigmoid = sigmoid<fp32, fp32>(_gate_upcast.y);
                gate_silu = _gate_upcast.y * gate_sigmoid;
                _up_grad.y = _output_grad_upcast.y * gate_silu;
                _gate_grad.y = _output_grad_upcast.y * _up_upcast.y * (gate_sigmoid + gate_silu * (1 - gate_sigmoid));

                up_grad_buffer[i] = dtype::reinterpret_2x16_as_32_bits(dtype::downcast(_up_grad));
                gate_grad_buffer[i] = dtype::reinterpret_2x16_as_32_bits(dtype::downcast(_gate_grad));
            }
        }

        ((fp32_4 *)gate_grad)[thread_id] = DType<fp32>::make4(gate_grad_buffer);
        ((fp32_4 *)up_grad)[thread_id] = DType<fp32>::make4(up_grad_buffer);
    } else if (start < num_elements) {
        // clang-format off
        #pragma unroll
        // clang-format on
        for (int i = start; i < num_elements; i++) {
            fp32 _up = dtype::upcast(static_cast<T>(up[i]));
            fp32 _gate = dtype::upcast(static_cast<T>(gate[i]));
            fp32 _output_grad = dtype::upcast(static_cast<T>(output_grad[i]));

            fp32 gate_sigmoid = sigmoid<fp32, fp32>(_gate);
            fp32 gate_silu = _gate * gate_sigmoid;

            up_grad[i] = _output_grad * gate_silu;
            gate_grad[i] = _output_grad * _up * (gate_sigmoid + gate_silu * (1 - gate_sigmoid));
        }
    }
}

void swiglu_forward_cuda(torch::Tensor gate, torch::Tensor up, torch::Tensor output, const int BLOCK_SIZE) {
    const int64_t num_elements = gate.numel();

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(
        gate.scalar_type(), "swiglu_forward_cuda_kernel", ([&] {
            const int vector_instruction_width = sizeof(fp32_4) / sizeof(scalar_t);

            const int num_elements_per_block = BLOCK_SIZE * vector_instruction_width;
            const int NUM_BLOCKS = (num_elements + num_elements_per_block - 1) / num_elements_per_block;

            _swiglu_forward_cuda_kernel<scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                gate.data_ptr<scalar_t>(), up.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), num_elements);
        }));
}

void swiglu_backward_cuda(torch::Tensor gate,
                          torch::Tensor up,
                          torch::Tensor output_grad,
                          torch::Tensor gate_grad,
                          torch::Tensor up_grad,
                          const int BLOCK_SIZE) {
    const int64_t num_elements = gate.numel();

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(
        gate.scalar_type(), "swiglu_backward_cuda_kernel", ([&] {
            const int vector_instruction_width = sizeof(fp32_4) / sizeof(scalar_t);

            const int num_elements_per_block = BLOCK_SIZE * vector_instruction_width;
            const int NUM_BLOCKS = (num_elements + num_elements_per_block - 1) / num_elements_per_block;

            _swiglu_backward_cuda_kernel<scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(gate.data_ptr<scalar_t>(),
                                                                               up.data_ptr<scalar_t>(),
                                                                               output_grad.data_ptr<scalar_t>(),
                                                                               gate_grad.data_ptr<scalar_t>(),
                                                                               up_grad.data_ptr<scalar_t>(),
                                                                               num_elements);
        }));
}
