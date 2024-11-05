#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

__device__ unsigned int get_threads_per_block() { return blockDim.x * blockDim.y * blockDim.z; }

__device__ unsigned int get_num_blocks() { return gridDim.x * gridDim.y * gridDim.z; }

__device__ unsigned int get_block_id() {
    return gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
}

__device__ int64_t get_local_thread_id() {
    return blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
}

__device__ int64_t get_global_thread_id() { return get_threads_per_block() * get_block_id() + get_local_thread_id(); }
