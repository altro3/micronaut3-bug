#include "varlen_embeddings.cuh"
#include "input_stage.h"
#include "data_types.h"

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
    void *stream_ptr,
    const bool dump_debug
) {
    if (total_tokens == 0 || out_features == 0 || threads_per_block <= 0) return;

    const int32_t blocks = total_tokens;
    const auto stream = static_cast<cudaStream_t>(stream_ptr);
    const auto type = static_cast<DataType>(data_type);
    __nv_bfloat16 *out_bf16 = static_cast<__nv_bfloat16 *>(out);

    switch (type) {
        case DataType::BF16:
            varlen_embeddings_fused_kernel<__nv_bfloat16><<<blocks, threads_per_block, 0, stream>>>(
                out_bf16, weight, weight_scales, tokens, seq_offsets, block_table, slot_mapping,
                max_blocks_per_seq, block_size, total_tokens, out_features, vocab_size, num_seqs, dump_debug
            );
            break;
        case DataType::FP8:
            varlen_embeddings_fused_kernel<__nv_fp8_e4m3><<<blocks, threads_per_block, 0, stream>>>(
                out_bf16, weight, weight_scales, tokens, seq_offsets, block_table, slot_mapping,
                max_blocks_per_seq, block_size, total_tokens, out_features, vocab_size, num_seqs, dump_debug
            );
            break;
        case DataType::FP4:
            varlen_embeddings_fused_kernel<__nv_fp4_e2m1><<<blocks, threads_per_block, 0, stream>>>(
                out_bf16, weight, weight_scales, tokens, seq_offsets, block_table, slot_mapping,
                max_blocks_per_seq, block_size, total_tokens, out_features, vocab_size, num_seqs, dump_debug
            );
            break;
    }
}
}
