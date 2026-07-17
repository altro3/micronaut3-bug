#ifndef FLASH_ATTENTION_TYPES_CUH
#define FLASH_ATTENTION_TYPES_CUH

#include <cuda_runtime.h>
#include <cuda_fp8.h>

#define WARP_SIZE 32

using ElementFP8 = __nv_fp8_e4m3;

__device__ __forceinline__ float warp_reduce_max_c(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum_c(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

#endif
