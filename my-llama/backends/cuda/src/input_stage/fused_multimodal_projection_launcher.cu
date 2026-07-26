#include "fused_multimodal_projection.cuh"
#include "data_types.h"
#include <cassert>
#include <cutlass/numeric_types.h>

void run_batched_projection_bf16(ProjectionParams params, const float *projection_bias, const float *quantization_scales, int32_t local_output_dim, int32_t input_feature_dim, int32_t rank_offset, dim3 grid, dim3 block, size_t shmem, cudaStream_t stream);

void run_batched_projection_fp8(ProjectionParams params, const float *projection_bias, const float *quantization_scales, int32_t local_output_dim, int32_t input_feature_dim, int32_t rank_offset, dim3 grid, dim3 block, size_t shmem, cudaStream_t stream);

void run_batched_projection_fp4(ProjectionParams params, const float *projection_bias, const float *quantization_scales, int32_t local_output_dim, int32_t input_feature_dim, int32_t rank_offset, dim3 grid, dim3 block, size_t shmem, cudaStream_t stream);

extern "C" void launch_fused_multimodal_projection(
    const void ** __restrict__ host_ptr_A,
    const void ** __restrict__ host_ptr_B,
    void ** __restrict__ host_ptr_D,
    const float * __restrict__ bias,
    const float * __restrict__ weight_scales,
    const int32_t * __restrict__ host_problem_shapes,
    const int32_t num_segments,
    const int32_t vision_hidden_size,
    const int32_t text_hidden_size,
    const int32_t tp_rank,
    const int32_t tp_size,
    int32_t data_type,
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
        max_m = m > max_m ? m : max_m;

        params.inputs[i] = host_ptr_A[i];
        params.weights[i] = host_ptr_B[i];
        params.outputs[i] = host_ptr_D[i];
        params.shapes[i] = cutlass::gemm::GemmCoord(m, local_output_dim, vision_hidden_size);
    }
    if (max_m <= 0) return;

    constexpr size_t shmem_load_size = 2 * (TILE_M * TILE_K + TILE_N * TILE_K) * sizeof(cutlass::bfloat16_t);
    constexpr size_t shmem_store_size = TILE_M * TILE_N * sizeof(float);
    constexpr size_t shmem_size = shmem_load_size > shmem_store_size ? shmem_load_size : shmem_store_size;

    constexpr dim3 block(128);
    const dim3 grid(
        (max_m + TILE_M - 1) / TILE_M,
        (local_output_dim + TILE_N - 1) / TILE_N,
        num_segments
    );

    switch (type) {
        case DataType::BF16:
            run_batched_projection_bf16(params, bias, weight_scales, local_output_dim, vision_hidden_size, rank_offset, grid, block, shmem_size, stream);
            break;
        case DataType::FP8:
            run_batched_projection_fp8(params, bias, weight_scales, local_output_dim, vision_hidden_size, rank_offset, grid, block, shmem_size, stream);
            break;
        case DataType::FP4:
            run_batched_projection_fp4(params, bias, weight_scales, local_output_dim, vision_hidden_size, rank_offset, grid, block, shmem_size, stream);
            break;
    }
}
