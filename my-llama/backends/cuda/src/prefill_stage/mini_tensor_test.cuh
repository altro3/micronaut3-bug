#pragma once
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t launch_mini_wmma_bf16(
    float *d_C,
    const void *d_A,
    const void *d_B,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif
