#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

#ifdef _WIN32
#ifdef EXPORT_KERNELS
#define KERNEL_API __declspec(dllexport)
#else
#define KERNEL_API __declspec(dllimport)
#endif
#else
#define KERNEL_API
#endif

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

extern "C" {
KERNEL_API void launch_fused_rmsnorm_forward(
    void *out,
    const void *input,
    const void *gamma,
    const float *gamma_scales,
    const float epsilon,
    const int32_t total_tokens,
    const int32_t hidden_size,
    int32_t data_type,
    const int32_t threads_per_block,
    void *stream_ptr
);
}