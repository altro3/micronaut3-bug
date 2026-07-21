#include "fused_multimodal_projection.cuh"
#include "data_types.h"
#include <cassert>
#include <cutlass/numeric_types.h>

extern "C" void launch_fused_multimodal_projection(
    const void ** __restrict__ host_ptr_A,
    const void ** __restrict__ host_ptr_B,
    void ** __restrict__ host_ptr_D,
    const float * __restrict__ bias,
    const float * __restrict__ weight_scales,
    const int32_t * __restrict__ host_problem_shapes,
    int32_t num_segments,
    int32_t vision_hidden_size,
    int32_t text_hidden_size,
    int32_t tp_rank,
    int32_t tp_size,
    int32_t data_type,
    void * __restrict__ workspace_ptr,
    void *stream_ptr
) {
    const auto stream = static_cast<cudaStream_t>(stream_ptr);
    const int32_t local_output_dim = text_hidden_size / tp_size;
    const int32_t rank_offset = tp_rank * local_output_dim;
    const auto type = static_cast<DataType>(data_type);

    assert(num_segments <= MAX_SEGMENTS);

    constexpr int32_t TILE_M = 64;
    constexpr int32_t TILE_N = 64;
    constexpr int32_t TILE_K = 32;

    int32_t max_m = 0;

    ProjectionParams params = {};

    for (int32_t i = 0; i < num_segments; ++i) {
        int32_t m = host_problem_shapes[i * 3 + 0];
        max_m = (m > max_m) ? m : max_m;

        params.inputs[i] = host_ptr_A[i];
        params.weights[i] = host_ptr_B[i];
        params.outputs[i] = host_ptr_D[i];
        params.shapes[i] = cutlass::gemm::GemmCoord(m, local_output_dim, vision_hidden_size);
    }
    if (max_m <= 0) return;

    constexpr size_t shmem_load_size = (TILE_M * TILE_K + TILE_N * TILE_K) * sizeof(cutlass::bfloat16_t);
    constexpr size_t shmem_store_size = TILE_M * TILE_N * sizeof(float);
    constexpr size_t shmem_size = shmem_load_size > shmem_store_size ? shmem_load_size : shmem_store_size;

    dim3 block(128);
    dim3 grid(
        (max_m + TILE_M - 1) / TILE_M,
        (local_output_dim + TILE_N - 1) / TILE_N,
        num_segments
    );

    switch (type) {
        case DataType::BF16:
            batched_projection_cutlass4_kernel<TILE_M, TILE_N, TILE_K, cutlass::bfloat16_t><<<grid, block, shmem_size, stream>>>(
                params, bias, weight_scales, local_output_dim, vision_hidden_size, rank_offset
            );
            break;
        case DataType::FP8:
            batched_projection_cutlass4_kernel<TILE_M, TILE_N, TILE_K, __nv_fp8_e4m3><<<grid, block, shmem_size, stream>>>(
                params, bias, weight_scales, local_output_dim, vision_hidden_size, rank_offset
            );
            break;
        case DataType::FP4:
            batched_projection_cutlass4_kernel<TILE_M, TILE_N, TILE_K, __nv_fp4_e2m1><<<grid, block, shmem_size, stream>>>(
                params, bias, weight_scales, local_output_dim, vision_hidden_size, rank_offset
            );
            break;
    }
}
