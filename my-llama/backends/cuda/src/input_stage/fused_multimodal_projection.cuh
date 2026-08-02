#pragma once
#include <stdint.h>

#include "core_api.h"
#include "cutlass/gemm/gemm.h"

constexpr int32_t MAX_SEGMENTS = 8;

struct ProjectionParams {
    const void *inputs[MAX_SEGMENTS];
    const void *weights[MAX_SEGMENTS];
    void *outputs[MAX_SEGMENTS];
    cutlass::gemm::GemmCoord shapes[MAX_SEGMENTS];
};

template<
    int TILE_M, int TILE_N, int TILE_K,
    typename T_Weight
>
__global__ void batched_projection_cutlass4_kernel(
    ProjectionParams params,
    const float * __restrict__ projection_bias,
    const float * __restrict__ quantization_scales,
    int32_t local_output_dim,
    int32_t input_feature_dim,
    int32_t rank_offset
);

extern "C" KERNEL_API void launch_fused_multimodal_projection(
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
    void *stream_ptr
);
