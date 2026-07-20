#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

template<typename T_weight>
__global__ void custom_projection_gemm_kernel(
    const __nv_bfloat16 * __restrict__ input_hidden_states,
    const void * __restrict__ projection_weights,
    __nv_bfloat16 * __restrict__ output_text_features,
    const float * __restrict__ projection_bias,
    const float * __restrict__ quantization_scales,
    int32_t batch_num_tokens,
    int32_t local_output_dim,
    int32_t input_feature_dim,
    int32_t rank_offset,
    int32_t tile_size_m,
    int32_t tile_size_n,
    int32_t tile_size_k
);
