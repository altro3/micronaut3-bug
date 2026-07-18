#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <stdint.h>

template<typename T>
struct VectorType;

template<>
struct VectorType<__nv_bfloat16> {
    using Type4 = uint4;
};

template<>
struct VectorType<__nv_fp8_e4m3> {
    using Type4 = uint4;
};

template<>
struct VectorType<__nv_fp4_e2m1> {
    using Type4 = uint4;
};

__device__ inline int32_t find_sequence_index(
    const int32_t * __restrict__ seq_offsets,
    const int32_t num_seqs,
    const int32_t global_token_idx
) {
    int32_t low = 0;
    int32_t high = num_seqs - 1;
    int32_t seq_idx = 0;

    while (low <= high) {
        const int32_t mid = low + (high - low) / 2;
        if (seq_offsets[mid] <= global_token_idx) {
            seq_idx = mid;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return seq_idx;
}

template<typename T>
__global__ void varlen_embeddings_fused_kernel(
    __nv_bfloat16 * __restrict__ out,
    const void * __restrict__ weight,
    const float * __restrict__ weight_scales,
    const uint32_t * __restrict__ tokens,
    const int32_t * __restrict__ seq_offsets,
    const int32_t * __restrict__ block_table,
    int32_t * __restrict__ slot_mapping,
    int32_t max_blocks_per_seq,
    int32_t block_size,
    int32_t total_tokens,
    int32_t out_features,
    int32_t vocab_size,
    int32_t num_seqs
);
