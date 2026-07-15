#include "cache_types.h"
#include <cuda_runtime.h>
#include <stdint.h>

template<CacheType T>
extern __global__ void paged_kv_cache_write_kernel(
    void * __restrict__ k_block_table,
    void * __restrict__ v_block_table,
    const float * __restrict__ k_src,
    const float * __restrict__ v_src,
    const int32_t * __restrict__ block_mapping,
    const int32_t * __restrict__ seq_lengths,
    float * __restrict__ k_scales,
    float * __restrict__ v_scales,
    int num_seqs,
    int num_kv_heads,
    int head_dim,
    int max_blocks_per_seq,
    int block_size,
    int is_prefill
);

template<CacheType T>
extern __global__ void paged_flash_decoding_partial_kernel(
    float * __restrict__ partial_out,
    float * __restrict__ partial_max,
    float * __restrict__ partial_sum,
    const float * __restrict__ query,
    const void * __restrict__ k_block_table,
    const void * __restrict__ v_block_table,
    const int32_t * __restrict__ block_mapping,
    const int32_t * __restrict__ seq_lengths,
    const float * __restrict__ k_scales,
    const float * __restrict__ v_scales,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_blocks_per_seq,
    int block_size,
    int chunk_size,
    int num_chunks
);

extern __global__ void paged_flash_decoding_combine_kernel(
    float * __restrict__ output,
    const float * __restrict__ partial_out,
    const float * __restrict__ partial_max,
    const float * __restrict__ partial_sum,
    int num_heads,
    int head_dim,
    int num_chunks
);

extern "C" {
void launch_paged_flash_decoding_write(
    void *k_block_table,
    void *v_block_table,
    const float *k_src,
    const float *v_src,
    const int32_t *block_mapping,
    const int32_t *seq_lengths,
    float *k_scales,
    float *v_scales,
    const int cache_type_id,
    const int num_seqs,
    const int num_kv_heads,
    const int head_dim,
    const int max_blocks_per_seq,
    const int block_size,
    const int is_prefill,
    void *stream_ptr
) {
    if (num_seqs == 0) return;

    const auto stream = static_cast<cudaStream_t>(stream_ptr);
    constexpr int threads = 128;
    dim3 blocks(num_seqs, num_kv_heads);

    switch (cache_type_id) {
        default:
        case 0:
            paged_kv_cache_write_kernel<CacheType::FP32><<<blocks, threads, 0, stream>>>(
                k_block_table, v_block_table, k_src, v_src, block_mapping, seq_lengths, k_scales, v_scales,
                num_seqs, num_kv_heads, head_dim, max_blocks_per_seq, block_size, is_prefill
            );
            break;
        case 1:
            paged_kv_cache_write_kernel<CacheType::FP16><<<blocks, threads, 0, stream>>>(
                k_block_table, v_block_table, k_src, v_src, block_mapping, seq_lengths, k_scales, v_scales,
                num_seqs, num_kv_heads, head_dim, max_blocks_per_seq, block_size, is_prefill
            );
            break;
        case 2:
            paged_kv_cache_write_kernel<CacheType::FP8><<<blocks, threads, 0, stream>>>(
                k_block_table, v_block_table, k_src, v_src, block_mapping, seq_lengths, k_scales, v_scales,
                num_seqs, num_kv_heads, head_dim, max_blocks_per_seq, block_size, is_prefill
            );
            break;
    }
}

void launch_paged_flash_decoding(
    float *output,
    float *partial_out,
    float *partial_max,
    float *partial_sum,
    const float *query,
    const void *k_block_table,
    const void *v_block_table,
    const int32_t *block_mapping,
    const int32_t *seq_lengths,
    const float *k_scales,
    const float *v_scales,
    const int cache_type_id,
    const int num_seqs,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int block_size,
    const int chunk_size,
    void *stream_ptr
) {
    if (num_seqs == 0) return;

    const auto stream = static_cast<cudaStream_t>(stream_ptr);
    const int num_chunks = (max_seq_len + chunk_size - 1) / chunk_size;

    constexpr int threads_p1 = 128;
    const size_t shared_mem_size = (head_dim + chunk_size) * sizeof(float);
    dim3 blocks_p1(num_heads, num_chunks, num_seqs);

    switch (cache_type_id) {
        default:
        case 0:
            paged_flash_decoding_partial_kernel<CacheType::FP32><<<blocks_p1, threads_p1, shared_mem_size, stream>>>(
                partial_out, partial_max, partial_sum, query, k_block_table, v_block_table, block_mapping, seq_lengths, k_scales, v_scales,
                num_heads, num_kv_heads, head_dim, max_blocks_per_seq, block_size, chunk_size, num_chunks
            );
            break;
        case 1:
            paged_flash_decoding_partial_kernel<CacheType::FP16><<<blocks_p1, threads_p1, shared_mem_size, stream>>>(
                partial_out, partial_max, partial_sum, query, k_block_table, v_block_table, block_mapping, seq_lengths, k_scales, v_scales,
                num_heads, num_kv_heads, head_dim, max_blocks_per_seq, block_size, chunk_size, num_chunks
            );
            break;
        case 2:
            paged_flash_decoding_partial_kernel<CacheType::FP8><<<blocks_p1, threads_p1, shared_mem_size, stream>>>(
                partial_out, partial_max, partial_sum, query, k_block_table, v_block_table, block_mapping, seq_lengths, k_scales, v_scales,
                num_heads, num_kv_heads, head_dim, max_blocks_per_seq, block_size, chunk_size, num_chunks
            );
            break;
    }

    constexpr int threads_p2 = 32;
    dim3 blocks_p2(num_heads, num_seqs);
    paged_flash_decoding_combine_kernel<<<blocks_p2, threads_p2, 0, stream>>>(
        output, partial_out, partial_max, partial_sum, num_heads, head_dim, num_chunks
    );
}
}
