#pragma once

#define WARP_SIZE 32
#define LOG_WARP_SIZE 5

#include <cuda.h>
#include <cuda_runtime.h>

#include "dtypes.h"

inline __device__ uint get_threads_per_block() { return blockDim.x * blockDim.y * blockDim.z; }

inline __device__ uint get_num_blocks() { return gridDim.x * gridDim.y * gridDim.z; }

inline __device__ uint get_block_id() {
    return gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
}

inline __device__ uint get_local_thread_id() {
    return blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
}

inline __device__ uint64 get_global_thread_id() {
    return get_threads_per_block() * get_block_id() + get_local_thread_id();
}
