#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void launch_varlen_embeddings(
    void *out,
    const void *weight,
    const float *weight_scales,
    const uint32_t *tokens,
    const int32_t *seq_offsets,
    const int32_t *block_table,
    int32_t *slot_mapping,
    int32_t max_blocks_per_seq,
    int32_t block_size,
    int32_t total_tokens,
    int32_t out_features,
    int32_t vocab_size,
    int32_t num_seqs,
    int32_t data_type,
    int32_t threads_per_block,
    void *stream_ptr
);

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
);

void launch_fused_multimodal_projection(
    void * __restrict__ out_tokens,
    const void * __restrict__ input_tokens,
    const void * __restrict__ weight_matrix,
    const float * __restrict__ bias,
    const int32_t * __restrict__ vision_segments,
    int32_t num_segments,
    int32_t vision_hidden_size,
    int32_t text_hidden_size,
    int32_t tp_rank,
    int32_t tp_size,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif
