#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <stdint.h>

#include "data_types.h"
#include "kernels.h"
#include "fused_rmsnorm_forward.cuh"

extern "C" {
void launch_fused_rmsnorm_forward(
    void *out,
    const void *input,
    const void *gamma,
    const float *gamma_scales,
    float epsilon,
    int32_t total_tokens,
    int32_t hidden_size,
    int32_t data_type,
    int32_t threads_per_block,
    void *stream_ptr
) {
    if (total_tokens <= 0 || hidden_size <= 0 || threads_per_block <= 0) return;

    const auto stream = static_cast<cudaStream_t>(stream_ptr);
    const auto type = static_cast<DataType>(data_type);
    __nv_bfloat16 *out_bf16 = static_cast<__nv_bfloat16 *>(out);

    const int32_t num_warps = threads_per_block / 32;
    const int32_t shared_mem_bytes = num_warps * sizeof(float);

    switch (type) {
        case DataType::BF16:
            fused_rmsnorm_forward_kernel<__nv_bfloat16><<<total_tokens, threads_per_block, shared_mem_bytes, stream>>>(
                out_bf16, input, gamma, gamma_scales, epsilon, total_tokens, hidden_size
            );
            break;
        case DataType::FP8:
            fused_rmsnorm_forward_kernel<__nv_fp8_e4m3><<<total_tokens, threads_per_block, shared_mem_bytes, stream>>>(
                out_bf16, input, gamma, gamma_scales, epsilon, total_tokens, hidden_size
            );
            break;
        case DataType::FP4:
            fused_rmsnorm_forward_kernel<__nv_fp4_e2m1><<<total_tokens, threads_per_block, shared_mem_bytes, stream>>>(
                out_bf16, input, gamma, gamma_scales, epsilon, total_tokens, hidden_size
            );
            break;
    }
}
}
