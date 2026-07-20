#include "fused_multimodal_projection.cuh"
#include "data_types.h"

#include "cutlass/gemm/gemm.h"

#include <cassert>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

extern "C" void launch_fused_multimodal_projection(
    const void ** __restrict__ host_ptr_A,
    const void ** __restrict__ host_ptr_B,
    void ** __restrict__ host_ptr_D,
    const float * __restrict__ bias,
    const float * __restrict__ weight_scales,
    const cutlass::gemm::GemmCoord * __restrict__ host_problem_shapes,
    const int32_t num_segments,
    const int32_t vision_hidden_size,
    const int32_t text_hidden_size,
    const int32_t tp_rank,
    const int32_t tp_size,
    int32_t data_type,
    void * __restrict__ workspace_ptr,
    const cudaStream_t stream
) {
    const int32_t local_output_dim = text_hidden_size / tp_size;
    const int32_t rank_offset = tp_rank * local_output_dim;
    const auto type = static_cast<DataType>(data_type);

    constexpr int32_t TILE_M = 64;
    constexpr int32_t TILE_N = 64;
    constexpr int32_t TILE_K = 32;

    constexpr size_t shmem_size = (TILE_M * TILE_K + TILE_N * TILE_K) * sizeof(__nv_bfloat16);

    for (int32_t i = 0; i < num_segments; ++i) {
        cutlass::gemm::GemmCoord problem_size = host_problem_shapes[i];
        if (problem_size.m() <= 0) continue;

        assert(problem_size.k() == vision_hidden_size);

        const int32_t batch_num_tokens = problem_size.m();
        const int32_t input_feature_dim = problem_size.k();

        if (workspace_ptr != nullptr) {
            constexpr uintptr_t align_mask = 15;
            const auto aligned_workspace = reinterpret_cast<void *>((reinterpret_cast<uintptr_t>(workspace_ptr) + align_mask) & ~align_mask);
            const auto sync_flag = static_cast<int32_t *>(aligned_workspace);
            cudaMemsetAsync(sync_flag, 0, sizeof(int32_t), stream);
        }

        dim3 block(128);
        dim3 grid((batch_num_tokens + (TILE_M - 1)) / TILE_M, (local_output_dim + (TILE_N - 1)) / TILE_N);

        const auto input_hidden_states = static_cast<const __nv_bfloat16 *>(host_ptr_A[i]);
        const void *projection_weights = host_ptr_B[i];
        const auto output_text_features = static_cast<__nv_bfloat16 *>(host_ptr_D[i]);
        const float *scale_ptr = (weight_scales != nullptr) ? (weight_scales + i) : nullptr;

        switch (type) {
            case DataType::BF16:
                custom_projection_gemm_kernel<__nv_bfloat16><<<grid, block, shmem_size, stream>>>(
                    input_hidden_states, projection_weights, output_text_features, bias, scale_ptr,
                    batch_num_tokens, local_output_dim, input_feature_dim, rank_offset, TILE_M, TILE_N, TILE_K
                );
                break;
            case DataType::FP8:
                custom_projection_gemm_kernel<__nv_fp8_e4m3><<<grid, block, shmem_size, stream>>>(
                    input_hidden_states, projection_weights, output_text_features, bias, scale_ptr,
                    batch_num_tokens, local_output_dim, input_feature_dim, rank_offset, TILE_M, TILE_N, TILE_K
                );
                break;
            case DataType::FP4:
                custom_projection_gemm_kernel<__nv_fp4_e2m1><<<grid, block, shmem_size, stream>>>(
                    input_hidden_states, projection_weights, output_text_features, bias, scale_ptr,
                    batch_num_tokens, local_output_dim, input_feature_dim, rank_offset, TILE_M, TILE_N, TILE_K
                );
                break;
        }
    }
}
