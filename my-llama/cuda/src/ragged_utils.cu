#include <cuda_runtime.h>
#include <stdint.h>

extern "C" {
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

__global__ void compute_slot_mapping_kernel(
    const int32_t * __restrict__ seq_offsets,
    const int32_t * __restrict__ block_table,
    const int32_t max_blocks_per_seq,
    const int32_t block_size,
    int32_t * __restrict__ slot_mapping,
    const int32_t total_batch_tokens,
    const int32_t num_seqs
) {
    const int global_token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_token_idx >= total_batch_tokens) return;

    const int32_t seq_idx = find_sequence_index(seq_offsets, num_seqs, global_token_idx);

    const int32_t start_tok_idx = seq_offsets[seq_idx];
    const int32_t token_local_idx = global_token_idx - start_tok_idx;

    const int32_t logical_block_idx = token_local_idx / block_size;
    const int32_t block_offset = token_local_idx % block_size;

    const int32_t *seq_blocks = block_table + seq_idx * max_blocks_per_seq;
    const int32_t physical_block_id = seq_blocks[logical_block_idx];

    const int32_t global_slot = physical_block_id * block_size + block_offset;

    slot_mapping[global_token_idx] = global_slot;
}

void launch_compute_slot_mapping(
    const int32_t *seq_offsets,
    const int32_t *block_table,
    const int32_t max_blocks_per_seq,
    const int32_t block_size,
    int32_t *slot_mapping,
    const int32_t total_batch_tokens,
    const int32_t num_seqs,
    const int32_t threads_per_block,
    cudaStream_t stream
) {
    if (total_batch_tokens == 0 || threads_per_block <= 0) return;

    const int blocks_per_grid = (total_batch_tokens + threads_per_block - 1) / threads_per_block;

    compute_slot_mapping_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        seq_offsets,
        block_table,
        max_blocks_per_seq,
        block_size,
        slot_mapping,
        total_batch_tokens,
        num_seqs
    );
}
}
