#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <algorithm>
#include "dequantize_fusion_mma_gguf_q4_k.cuh"
#include "data_types.h"

void run_fused_gemm_gguf_q4_k_bf16(cutlass::bfloat16_t *output, const cutlass::bfloat16_t *input_A, const BlockQ4K *input_B_quant, int32_t M, int32_t N, int32_t K, dim3 grid, dim3 block, size_t shmem, cudaStream_t stream);

void run_fused_gemm_gguf_q4_k_fp8(__nv_fp8_e4m3 *output, const __nv_fp8_e4m3 *input_A, const BlockQ4K *input_B_quant, int32_t M, int32_t N, int32_t K, dim3 grid, dim3 block, size_t shmem, cudaStream_t stream);

void run_fused_gemm_gguf_q4_k_fp4(__nv_fp4_e2m1 *output, const __nv_fp4_e2m1 *input_A, const BlockQ4K *input_B_quant, int32_t M, int32_t N, int32_t K, dim3 grid, dim3 block, size_t shmem, cudaStream_t stream);

static size_t get_shmem_size_dynamic(int32_t TILE_M, int32_t TILE_N, int32_t TILE_K, DataType type) {
    size_t element_size_A = 2;
    if (type == DataType::FP8) element_size_A = 1;
    if (type == DataType::FP4) element_size_A = 1;

    const size_t shmem_input = TILE_M * TILE_K * element_size_A + TILE_N * TILE_K * sizeof(__nv_bfloat16);
    const size_t shmem_output = TILE_M * TILE_N * sizeof(float);
    return std::max(shmem_input, shmem_output);
}

extern "C" void launch_fused_gemm_gguf_q4_k(
    void *output_activations,
    const void *input_activations,
    const void *quantized_weights,
    const int32_t batch_size_or_tokens,
    const int32_t hidden_units_out,
    const int32_t hidden_units_in,
    int32_t data_type,
    void *stream_ptr
) {
    const auto stream = static_cast<cudaStream_t>(stream_ptr);
    const auto type = static_cast<DataType>(data_type);

    constexpr int32_t TILE_M = 64;
    constexpr int32_t TILE_N = 64;
    constexpr int32_t TILE_K = 32;

    constexpr dim3 block(128);
    const dim3 grid((batch_size_or_tokens + TILE_M - 1) / TILE_M, (hidden_units_out + TILE_N - 1) / TILE_N, 1);

    const size_t shmem_size = get_shmem_size_dynamic(TILE_M, TILE_N, TILE_K, type);

    switch (type) {
        case DataType::BF16: {
            if (shmem_size >= 48 * 1024) {
                cudaFuncSetAttribute(reinterpret_cast<const void *>(run_fused_gemm_gguf_q4_k_bf16), cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
            }
            run_fused_gemm_gguf_q4_k_bf16(
                static_cast<cutlass::bfloat16_t *>(output_activations),
                static_cast<const cutlass::bfloat16_t *>(input_activations),
                static_cast<const BlockQ4K *>(quantized_weights),
                batch_size_or_tokens, hidden_units_out, hidden_units_in, grid, block, shmem_size, stream
            );
            break;
        }
        case DataType::FP8: {
            if (shmem_size >= 48 * 1024) {
                cudaFuncSetAttribute(reinterpret_cast<const void *>(run_fused_gemm_gguf_q4_k_fp8), cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
            }
            run_fused_gemm_gguf_q4_k_fp8(
                static_cast<__nv_fp8_e4m3 *>(output_activations),
                static_cast<const __nv_fp8_e4m3 *>(input_activations),
                static_cast<const BlockQ4K *>(quantized_weights),
                batch_size_or_tokens, hidden_units_out, hidden_units_in, grid, block, shmem_size, stream
            );
            break;
        }
        case DataType::FP4: {
            if (shmem_size >= 48 * 1024) {
                cudaFuncSetAttribute(reinterpret_cast<const void *>(run_fused_gemm_gguf_q4_k_fp4), cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
            }
            run_fused_gemm_gguf_q4_k_fp4(
                static_cast<__nv_fp4_e2m1 *>(output_activations),
                static_cast<const __nv_fp4_e2m1 *>(input_activations),
                static_cast<const BlockQ4K *>(quantized_weights),
                batch_size_or_tokens, hidden_units_out, hidden_units_in, grid, block, shmem_size, stream
            );
            break;
        }
    }
}
