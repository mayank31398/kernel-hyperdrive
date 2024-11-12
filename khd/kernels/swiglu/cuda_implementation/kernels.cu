#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../../utils/activations.h"
#include "../../utils/dtypes.h"
#include "../../utils/threads.h"

template <typename scalar_t, typename vector_t>
__global__ void _swiglu_forward_cuda_kernel(const scalar_t *gate,
                                            const scalar_t *up,
                                            scalar_t *output,
                                            const int64_t num_elements) {
    constexpr int vector_instruction_width = sizeof(vector_t) / sizeof(scalar_t);
    const int64_t thread_id = get_global_thread_id();

    using dtype = DType<scalar_t>;

    if constexpr (vector_instruction_width == 1) {
        if (thread_id < num_elements) {
            fp32 _gate_upcast = dtype::upcast(gate[thread_id]);

            // up is upcasted automatically
            _gate_upcast = up[thread_id] * _gate_upcast * sigmoid<fp32, fp32>(_gate_upcast);
            output[thread_id] = dtype::downcast(_gate_upcast);
        }
    } else {
        const int64_t start = thread_id * vector_instruction_width;
        const int64_t end = (thread_id + 1) * vector_instruction_width - 1;  // inclusive of last element

        if (start < num_elements && end < num_elements) {
            vector_t *output_vec = (vector_t *)output;

            if constexpr (std::is_same_v<scalar_t, fp32>) {
                const fp32 *gate_vec = (fp32 *)&((vector_t *)gate)[thread_id];
                const fp32 *up_vec = (fp32 *)&((vector_t *)up)[thread_id];
                fp32 output_buffer[vector_instruction_width];

                // clang-format off
                #pragma unroll
                // clang-format on
                for (int i = 0; i < vector_instruction_width; i++) {
                    output_buffer[i] = up_vec[i] * gate_vec[i] * sigmoid<fp32, fp32>(gate_vec[i]);
                }

                if constexpr (vector_instruction_width == 2) {
                    output_vec[thread_id] = dtype::make2(output_buffer);
                } else if constexpr (vector_instruction_width == 4) {
                    output_vec[thread_id] = dtype::make4(output_buffer);
                }
            } else {
                using T2 = typename dtype::nv_dtype2;

                if constexpr (vector_instruction_width == 2) {
                    T2 _gate = ((vector_t *)gate)[thread_id];
                    T2 _up = ((vector_t *)up)[thread_id];

                    fp32_2 _gate_upcast = dtype::upcast(_gate);
                    fp32_2 _up_upcast = dtype::upcast(_up);

                    _gate_upcast =
                        DType<fp32>::make2(_up_upcast.x * _gate_upcast.x * sigmoid<fp32, fp32>(_gate_upcast.x),
                                           _up_upcast.y * _gate_upcast.y * sigmoid<fp32, fp32>(_gate_upcast.y));

                    output_vec[thread_id] = dtype::downcast(_gate_upcast);
                } else {
                    const fp32 *gate_vec = (fp32 *)&((vector_t *)gate)[thread_id];
                    const fp32 *up_vec = (fp32 *)&((vector_t *)up)[thread_id];

                    const int n = vector_instruction_width >> 1;
                    fp32 output_buffer[n];

                    // clang-format off
                    #pragma unroll
                    // clang-format on
                    for (int i = 0; i < n; i++) {
                        fp32_2 _gate_upcast = dtype::upcast(dtype::reinterpret_32_bits_as_2x16(gate_vec[i]));
                        fp32_2 _up_upcast = dtype::upcast(dtype::reinterpret_32_bits_as_2x16(up_vec[i]));

                        _gate_upcast =
                            DType<fp32>::make2(_up_upcast.x * _gate_upcast.x * sigmoid<fp32, fp32>(_gate_upcast.x),
                                               _up_upcast.y * _gate_upcast.y * sigmoid<fp32, fp32>(_gate_upcast.y));

                        output_buffer[i] = dtype::reinterpret_2x16_as_32_bits(dtype::downcast(_gate_upcast));
                    }

                    if constexpr (vector_instruction_width == 4) {
                        output_vec[thread_id] = DType<fp32>::make2(output_buffer);
                    } else if constexpr (vector_instruction_width == 8) {
                        output_vec[thread_id] = DType<fp32>::make4(output_buffer);
                    }
                }
            }
        } else if (start < num_elements) {
            // clang-format off
            #pragma unroll
            // clang-format on
            for (int64_t i = start; i < num_elements; i++) {
                fp32 _gate_upcast = dtype::upcast(gate[i]);

                // up is upcasted automatically
                _gate_upcast = up[i] * _gate_upcast * sigmoid<fp32, fp32>(_gate_upcast);
                output[i] = dtype::downcast(_gate_upcast);
            }
        }
    }
}

__device__ void _swiglu_backward_helper_bf16_fp16_vectorized(fp32_2 &_gate_upcast, fp32_2 &_up_upcast) {
    fp32 _gate_sigmoid_x = sigmoid<fp32, fp32>(_gate_upcast.x);
    fp32 _gate_sigmoid_y = sigmoid<fp32, fp32>(_gate_upcast.y);

    fp32 _gate_silu_x = _gate_upcast.x * _gate_sigmoid_x;
    fp32 _gate_silu_y = _gate_upcast.y * _gate_sigmoid_y;

    _gate_upcast = DType<fp32>::make2(
        _output_grad_upcast.x * _up_upcast.x * (_gate_sigmoid_x + _gate_silu_x * (1 - _gate_sigmoid_x)),
        _output_grad_upcast.y * _up_upcast.y * (_gate_sigmoid_y + _gate_silu_y * (1 - _gate_sigmoid_y)));

    _up_upcast = DType<fp32>::make2(_output_grad_upcast.x * _gate_silu_x, _output_grad_upcast.y * _gate_silu_y);
}

template <typename scalar_t, typename vector_t>
__global__ void _swiglu_backward_cuda_kernel(const scalar_t *gate,
                                             const scalar_t *up,
                                             const scalar_t *output_grad,
                                             scalar_t *gate_grad,
                                             scalar_t *up_grad,
                                             const int64_t num_elements) {
    constexpr int vector_instruction_width = sizeof(vector_t) / sizeof(scalar_t);
    const int64_t thread_id = get_global_thread_id();

    using dtype = DType<scalar_t>;

    if constexpr (vector_instruction_width == 1) {
        if (thread_id < num_elements) {
            fp32 _gate_upcast = dtype::upcast(gate[thread_id]);

            fp32 _gate_sigmoid = sigmoid<fp32, fp32>(_gate_upcast);
            fp32 _gate_silu = _gate_upcast * _gate_sigmoid;

            gate_grad[thread_id] = dtype::downcast(output_grad[thread_id] * up[thread_id] *
                                                   (_gate_sigmoid + _gate_silu * (1 - _gate_sigmoid)));
            up_grad[thread_id] = dtype::downcast(output_grad[thread_id] * _gate_silu);
        }
    } else {
        using T = typename dtype::nv_dtype;
        using T2 = typename dtype::nv_dtype2;

        const int64_t start = thread_id * vector_instruction_width;
        const int64_t end = (thread_id + 1) * vector_instruction_width - 1;  // inclusive of last element

        if (start < num_elements && end < num_elements) {
            vector_t *gate_grad_vec = (vector_t *)gate_grad;
            vector_t *up_grad_vec = (vector_t *)up_grad;

            if constexpr (std::is_same_v<scalar_t, fp32>) {
                const fp32 *gate_vec = (fp32 *)&((vector_t *)gate)[thread_id];
                const fp32 *up_vec = (fp32 *)&((vector_t *)up)[thread_id];
                const fp32 *output_grad_vec = (fp32 *)&((vector_t *)output_grad)[thread_id];

                fp32 gate_grad_buffer[vector_instruction_width];
                fp32 up_grad_buffer[vector_instruction_width];

                // clang-format off
                #pragma unroll
                // clang-format on
                for (int i = 0; i < vector_instruction_width; i++) {
                    fp32 _gate_sigmoid = sigmoid<fp32, fp32>(gate_vec[i]);
                    fp32 _gate_silu = gate_vec[i] * _gate_sigmoid;

                    gate_grad_buffer[i] =
                        output_grad_vec[i] * up_vec[i] * (_gate_sigmoid + _gate_silu * (1 - _gate_sigmoid));
                    up_grad_buffer[i] = output_grad_vec[i] * _gate_silu;
                }

                if constexpr (vector_instruction_width == 2) {
                    gate_grad_vec[thread_id] = dtype::make2(gate_grad_buffer);
                    up_grad_vec[thread_id] = dtype::make2(up_grad_buffer);
                } else if constexpr (vector_instruction_width == 4) {
                    gate_grad_vec[thread_id] = dtype::make4(gate_grad_buffer);
                    up_grad_vec[thread_id] = dtype::make4(up_grad_buffer);
                }
            } else {
                using T2 = typename dtype::nv_dtype2;

                if constexpr (vector_instruction_width == 2) {
                    T2 _gate = ((vector_t *)gate)[thread_id];
                    T2 _up = ((vector_t *)up)[thread_id];
                    T2 _output_grad = ((vector_t *)output_grad)[thread_id];

                    fp32_2 _gate_upcast = dtype::upcast(_gate);
                    fp32_2 _up_upcast = dtype::upcast(_up);
                    fp32_2 _output_grad_upcast = dtype::upcast(_output_grad);

                    _swiglu_backward_helper_bf16_fp16_vectorized(_gate_upcast, _up_upcast);

                    gate_grad_vec[thread_id] = dtype::downcast(_gate_upcast);
                    up_grad_vec[thread_id] = dtype::downcast(_up_upcast);
                } else {
                    const fp32 *gate_vec = (fp32 *)&((vector_t *)gate)[thread_id];
                    const fp32 *up_vec = (fp32 *)&((vector_t *)up)[thread_id];
                    const fp32 *output_grad_vec = (fp32 *)&((vector_t *)output_grad)[thread_id];

                    const int n = vector_instruction_width >> 1;
                    fp32 gate_grad_buffer[n];
                    fp32 up_grad_buffer[n];

                    // clang-format off
                    #pragma unroll
                    // clang-format on
                    for (int i = 0; i < n; i++) {
                        fp32_2 _gate_upcast = dtype::upcast(dtype::reinterpret_32_bits_as_2x16(gate_vec[i]));
                        fp32_2 _up_upcast = dtype::upcast(dtype::reinterpret_32_bits_as_2x16(up_vec[i]));
                        fp32_2 _output_grad_upcast =
                            dtype::upcast(dtype::reinterpret_32_bits_as_2x16(output_grad_vec[i]));

                        _swiglu_backward_helper_bf16_fp16_vectorized(_gate_upcast, _up_upcast);

                        gate_grad_buffer[i] = dtype::reinterpret_2x16_as_32_bits(dtype::downcast(_gate_upcast));
                        up_grad_buffer[i] = dtype::reinterpret_2x16_as_32_bits(dtype::downcast(_up_upcast));
                    }

                    if constexpr (vector_instruction_width == 4) {
                        gate_grad_vec[thread_id] = DType<fp32>::make2(gate_grad_buffer);
                        up_grad_vec[thread_id] = DType<fp32>::make2(up_grad_buffer);
                    } else if constexpr (vector_instruction_width == 8) {
                        gate_grad_vec[thread_id] = DType<fp32>::make4(gate_grad_buffer);
                        up_grad_vec[thread_id] = DType<fp32>::make4(up_grad_buffer);
                    }
                }
            }
        } else if (start < num_elements) {
            // clang-format off
            #pragma unroll
            // clang-format on
            for (int i = start; i < num_elements; i++) {
                fp32 _gate_upcast = dtype::upcast(gate[i]);

                fp32 _gate_sigmoid = sigmoid<fp32, fp32>(_gate_upcast);
                fp32 _gate_silu = _gate_upcast * _gate_sigmoid;

                gate_grad[i] =
                    dtype::downcast(output_grad[i] * up[i] * (_gate_sigmoid + _gate_silu * (1 - _gate_sigmoid)));
                up_grad[i] = dtype::downcast(output_grad[i] * _gate_silu);
            }
        }
    }
}

void swiglu_forward_cuda(const torch::Tensor &gate,
                         const torch::Tensor &up,
                         torch::Tensor output,
                         const int &vector_instruction_width,
                         const int &BLOCK_SIZE) {
    const int64_t num_elements = gate.numel();

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(
        gate.scalar_type(), "swiglu_forward_cuda_kernel", ([&] {
            const int num_elements_per_block = BLOCK_SIZE * vector_instruction_width;
            const int NUM_BLOCKS = (num_elements + num_elements_per_block - 1) / num_elements_per_block;

            switch (vector_instruction_width) {
                case 1:
                    _swiglu_forward_cuda_kernel<scalar_t, scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                        gate.data_ptr<scalar_t>(), up.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), num_elements);
                    break;
                case 2:
                    using vector_t = typename DType<scalar_t>::nv_dtype2;
                    _swiglu_forward_cuda_kernel<scalar_t, vector_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                        gate.data_ptr<scalar_t>(), up.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), num_elements);
                    break;
                case 4:
                    if constexpr (std::is_same_v<scalar_t, fp32>) {
                        _swiglu_forward_cuda_kernel<scalar_t, fp32_4>
                            <<<NUM_BLOCKS, BLOCK_SIZE>>>(gate.data_ptr<scalar_t>(),
                                                         up.data_ptr<scalar_t>(),
                                                         output.data_ptr<scalar_t>(),
                                                         num_elements);
                    } else {
                        _swiglu_forward_cuda_kernel<scalar_t, fp32_2>
                            <<<NUM_BLOCKS, BLOCK_SIZE>>>(gate.data_ptr<scalar_t>(),
                                                         up.data_ptr<scalar_t>(),
                                                         output.data_ptr<scalar_t>(),
                                                         num_elements);
                    }
                    break;
                case 8:
                    if constexpr (std::is_same_v<scalar_t, fp32>) {
                        throw std::runtime_error("fp32 doesn't support vector_instruction_width = 8");
                    } else {
                        _swiglu_forward_cuda_kernel<scalar_t, fp32_4>
                            <<<NUM_BLOCKS, BLOCK_SIZE>>>(gate.data_ptr<scalar_t>(),
                                                         up.data_ptr<scalar_t>(),
                                                         output.data_ptr<scalar_t>(),
                                                         num_elements);
                    }
                    break;
                default:
                    throw std::runtime_error("invalid vector_instruction_width");
                    break;
            }
        }));
}

void swiglu_backward_cuda(const torch::Tensor &gate,
                          const torch::Tensor &up,
                          const torch::Tensor &output_grad,
                          torch::Tensor gate_grad,
                          torch::Tensor up_grad,
                          const int &vector_instruction_width,
                          const int &BLOCK_SIZE) {
    const int64_t num_elements = gate.numel();

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(
        gate.scalar_type(), "swiglu_backward_cuda_kernel", ([&] {
            const int num_elements_per_block = BLOCK_SIZE * vector_instruction_width;
            const int NUM_BLOCKS = (num_elements + num_elements_per_block - 1) / num_elements_per_block;

            switch (vector_instruction_width) {
                case 1:
                    _swiglu_backward_cuda_kernel<scalar_t, scalar_t>
                        <<<NUM_BLOCKS, BLOCK_SIZE>>>(gate.data_ptr<scalar_t>(),
                                                     up.data_ptr<scalar_t>(),
                                                     output_grad.data_ptr<scalar_t>(),
                                                     gate_grad.data_ptr<scalar_t>(),
                                                     up_grad.data_ptr<scalar_t>(),
                                                     num_elements);
                    break;
                case 2:
                    using vector_t = typename DType<scalar_t>::nv_dtype2;
                    _swiglu_backward_cuda_kernel<scalar_t, vector_t>
                        <<<NUM_BLOCKS, BLOCK_SIZE>>>(gate.data_ptr<scalar_t>(),
                                                     up.data_ptr<scalar_t>(),
                                                     output_grad.data_ptr<scalar_t>(),
                                                     gate_grad.data_ptr<scalar_t>(),
                                                     up_grad.data_ptr<scalar_t>(),
                                                     num_elements);
                    break;
                case 4:
                    if constexpr (std::is_same_v<scalar_t, fp32>) {
                        _swiglu_backward_cuda_kernel<scalar_t, fp32_4>
                            <<<NUM_BLOCKS, BLOCK_SIZE>>>(gate.data_ptr<scalar_t>(),
                                                         up.data_ptr<scalar_t>(),
                                                         output_grad.data_ptr<scalar_t>(),
                                                         gate_grad.data_ptr<scalar_t>(),
                                                         up_grad.data_ptr<scalar_t>(),
                                                         num_elements);
                    } else {
                        _swiglu_backward_cuda_kernel<scalar_t, fp32_2>
                            <<<NUM_BLOCKS, BLOCK_SIZE>>>(gate.data_ptr<scalar_t>(),
                                                         up.data_ptr<scalar_t>(),
                                                         output_grad.data_ptr<scalar_t>(),
                                                         gate_grad.data_ptr<scalar_t>(),
                                                         up_grad.data_ptr<scalar_t>(),
                                                         num_elements);
                    }
                    break;
                case 8:
                    if constexpr (std::is_same_v<scalar_t, fp32>) {
                        throw std::runtime_error("fp32 doesn't support vector_instruction_width = 8");
                    } else {
                        _swiglu_backward_cuda_kernel<scalar_t, fp32_4>
                            <<<NUM_BLOCKS, BLOCK_SIZE>>>(gate.data_ptr<scalar_t>(),
                                                         up.data_ptr<scalar_t>(),
                                                         output_grad.data_ptr<scalar_t>(),
                                                         gate_grad.data_ptr<scalar_t>(),
                                                         up_grad.data_ptr<scalar_t>(),
                                                         num_elements);
                    }
                    break;
                default:
                    throw std::runtime_error("invalid vector_instruction_width");
                    break;
            }
        }));
}
