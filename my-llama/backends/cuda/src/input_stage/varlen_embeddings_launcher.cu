#include "varlen_embeddings.cuh"
#include "input_stage.h"
#include "data_types.h"

void run_varlen_embeddings_bf16(__nv_bfloat16 *out, const void *weight, const float *weight_scales, const uint32_t *tokens, const int32_t *seq_offsets, const int32_t *block_table, int32_t *slot_mapping, int32_t max_blocks, int32_t b_size, int32_t t_tokens, int32_t out_f, int32_t v_size, int32_t n_seqs, int32_t tpb, cudaStream_t stream);

void run_varlen_embeddings_fp8(__nv_bfloat16 *out, const void *weight, const float *weight_scales, const uint32_t *tokens, const int32_t *seq_offsets, const int32_t *block_table, int32_t *slot_mapping, int32_t max_blocks, int32_t b_size, int32_t t_tokens, int32_t out_f, int32_t v_size, int32_t n_seqs, int32_t tpb, cudaStream_t stream);

void run_varlen_embeddings_fp4(__nv_bfloat16 *out, const void *weight, const float *weight_scales, const uint32_t *tokens, const int32_t *seq_offsets, const int32_t *block_table, int32_t *slot_mapping, int32_t max_blocks, int32_t b_size, int32_t t_tokens, int32_t out_f, int32_t v_size, int32_t n_seqs, int32_t tpb, cudaStream_t stream);

extern "C" {
void launch_varlen_embeddings(
    void *out,
    const void *weight,
    const float *weight_scales,
    const uint32_t *tokens,
    const int32_t *seq_offsets,
    const int32_t *block_table,
    int32_t *slot_mapping,
    const int32_t max_blocks_per_seq,
    const int32_t block_size,
    const int32_t total_tokens,
    const int32_t out_features,
    const int32_t vocab_size,
    const int32_t num_seqs,
    const int32_t data_type,
    const int32_t threads_per_block,
    void *stream_ptr
) {
    if (total_tokens == 0 || out_features == 0 || threads_per_block <= 0) return;

    const auto stream = static_cast<cudaStream_t>(stream_ptr);
    const auto type = static_cast<DataType>(data_type);
    __nv_bfloat16 *out_bf16 = static_cast<__nv_bfloat16 *>(out);
    switch (type) {
        case DataType::BF16:
            run_varlen_embeddings_bf16(out_bf16, weight, weight_scales, tokens, seq_offsets, block_table, slot_mapping, max_blocks_per_seq, block_size, total_tokens, out_features, vocab_size, num_seqs, threads_per_block, stream);
            break;
        case DataType::FP8:
            run_varlen_embeddings_fp8(out_bf16, weight, weight_scales, tokens, seq_offsets, block_table, slot_mapping, max_blocks_per_seq, block_size, total_tokens, out_features, vocab_size, num_seqs, threads_per_block, stream);
            break;
        case DataType::FP4:
            run_varlen_embeddings_fp4(out_bf16, weight, weight_scales, tokens, seq_offsets, block_table, slot_mapping, max_blocks_per_seq, block_size, total_tokens, out_features, vocab_size, num_seqs, threads_per_block, stream);
            break;
    }
}
}
