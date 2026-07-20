#pragma once
#include <stdint.h>
#include "cutlass/gemm/gemm.h"

template<typename T_weight>
__global__ void batched_projection_gemm_kernel(
    const void ** __restrict__ device_table_A,
    const void ** __restrict__ device_table_B,
    void ** __restrict__ device_table_D,
    const cutlass::gemm::GemmCoord * __restrict__ device_shapes,
    const float * __restrict__ projection_bias,
    const float * __restrict__ quantization_scales,
    int32_t local_output_dim,
    int32_t input_feature_dim,
    int32_t rank_offset,
    int32_t tile_size_m,
    int32_t tile_size_n,
    int32_t tile_size_k
);
