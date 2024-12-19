#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../../../include/activations.h"
#include "../../../include/dtypes/all.h"
#include "../../../include/threads.h"

template <typename scalar_t, typename vector_t>
__global__ void _swiglu_forward_cuda_kernel(const scalar_t *gate,
                                            const scalar_t *up,
                                            scalar_t *output,
                                            const int64_t num_elements) {
    constexpr int vector_instruction_width = sizeof(vector_t) / sizeof(scalar_t);
    static_assert(vector_instruction_width == 1 || vector_instruction_width == 2 || vector_instruction_width == 4 ||
                  vector_instruction_width == 8);

    const uint64 thread_id = get_global_thread_id();
    using dtype = DType<scalar_t>;

    if constexpr (vector_instruction_width == 1) {
        if (thread_id < num_elements) {
            fp32 _gate_upcast = dtype::upcast(gate[thread_id]);

            // up is upcasted automatically
            _gate_upcast = up[thread_id] * _gate_upcast * sigmoid<fp32, fp32>(_gate_upcast);
            output[thread_id] = dtype::downcast(_gate_upcast);
        }
    } else {
        int64_t end = (thread_id + 1) * vector_instruction_width - 1;  // inclusive of last element

        if (end < num_elements) {
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
                } else {
                    static_assert("vector_instruction_width is invalid for fp32");
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
                    } else {
                        static_assert("vector_instruction_width is invalid for fp16 & bf16");
                    }
                }
            }
        }

        // use first warp for computing the last elements
        if (thread_id < WARP_SIZE) {
            // NOTE end is same as start since we don't use vector load stores here
            end = (num_elements / vector_instruction_width) * vector_instruction_width + thread_id;
            if (end < num_elements) {
                fp32 _gate_upcast = dtype::upcast(gate[end]);

                // up is upcasted automatically
                _gate_upcast = up[end] * _gate_upcast * sigmoid<fp32, fp32>(_gate_upcast);
                output[end] = dtype::downcast(_gate_upcast);
            }
        }
    }
}

void swiglu_forward_cuda(const torch::Tensor &gate,
                         const torch::Tensor &up,
                         torch::Tensor &output,
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
