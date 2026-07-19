#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

template<typename T>
__global__ void fused_rmsnorm_forward_kernel(
    __nv_bfloat16 * __restrict__ out,
    const void * __restrict__ input,
    const void * __restrict__ gamma,
    const float * __restrict__ gamma_scales,
    float epsilon,
    int32_t total_tokens,
    int32_t hidden_size
);
