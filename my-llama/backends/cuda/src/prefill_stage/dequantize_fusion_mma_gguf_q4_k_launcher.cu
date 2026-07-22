#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <algorithm>
#include "dequantize_fusion_mma_gguf_q4_k.cuh"
#include "data_types.h"

template<typename ElementAct, int TILE_M, int TILE_N, int TILE_K>
extern __global__ void fused_gemm_gguf_q4_k_kernel(
    ElementAct * __restrict__ output_activations,
    const ElementAct * __restrict__ input_activations,
    const BlockQ4K * __restrict__ quantized_weights,
    int32_t batch_size_or_tokens, int32_t hidden_units_out, int32_t hidden_units_in
);

extern "C" void launch_fused_gemm_gguf_q4_k(
    void *output_activations,
    const void *input_activations,
    const void *quantized_weights,
    int32_t batch_size_or_tokens, int32_t hidden_units_out, int32_t hidden_units_in,
    int32_t data_type,
    void *stream_ptr
) {
    const auto stream = static_cast<cudaStream_t>(stream_ptr);
    const auto type = static_cast<DataType>(data_type);

    constexpr int32_t TILE_M = 64;
    constexpr int32_t TILE_N = 64;
    constexpr int32_t TILE_K = 32;

    dim3 block(128);
    dim3 grid(
        (batch_size_or_tokens + TILE_M - 1) / TILE_M,
        (hidden_units_out + TILE_N - 1) / TILE_N,
        1
    );

    size_t shmem_input = (TILE_M * TILE_K + TILE_N * TILE_K) * sizeof(cutlass::bfloat16_t);
    size_t shmem_output = (TILE_M * TILE_N) * sizeof(float);
    size_t shmem_size = std::max(shmem_input, shmem_output);

    switch (type) {
        case DataType::BF16: {
            auto kernel_ptr = fused_gemm_gguf_q4_k_kernel<cutlass::bfloat16_t, TILE_M, TILE_N, TILE_K>;
            if (shmem_size >= 48 * 1024) {
                cudaFuncSetAttribute(reinterpret_cast<const void *>(kernel_ptr), cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
            }
            kernel_ptr<<<grid, block, shmem_size, stream>>>(
                static_cast<cutlass::bfloat16_t *>(output_activations),
                static_cast<const cutlass::bfloat16_t *>(input_activations),
                static_cast<const BlockQ4K *>(quantized_weights),
                batch_size_or_tokens, hidden_units_out, hidden_units_in
            );
            break;
        }
        case DataType::FP8: {
            auto kernel_ptr = fused_gemm_gguf_q4_k_kernel<__nv_fp8_e4m3, TILE_M, TILE_N, TILE_K>;
            if (shmem_size >= 48 * 1024) {
                cudaFuncSetAttribute(reinterpret_cast<const void *>(kernel_ptr), cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
            }
            kernel_ptr<<<grid, block, shmem_size, stream>>>(
                static_cast<__nv_fp8_e4m3 *>(output_activations),
                static_cast<const __nv_fp8_e4m3 *>(input_activations),
                static_cast<const BlockQ4K *>(quantized_weights),
                batch_size_or_tokens, hidden_units_out, hidden_units_in
            );
            break;
        }
        case DataType::FP4: {
            auto kernel_ptr = fused_gemm_gguf_q4_k_kernel<__nv_fp4_e2m1, TILE_M, TILE_N, TILE_K>;
            if (shmem_size >= 48 * 1024) {
                cudaFuncSetAttribute(reinterpret_cast<const void *>(kernel_ptr), cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
            }
            kernel_ptr<<<grid, block, shmem_size, stream>>>(
                static_cast<__nv_fp4_e2m1 *>(output_activations),
                static_cast<const __nv_fp4_e2m1 *>(input_activations),
                static_cast<const BlockQ4K *>(quantized_weights),
                batch_size_or_tokens, hidden_units_out, hidden_units_in
            );
            break;
        }
    }
}
