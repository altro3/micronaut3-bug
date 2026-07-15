#include <cuda_runtime.h>

extern __global__ void flash_decoding_partial_kernel(
    float * __restrict__ partial_out,
    float * __restrict__ partial_max,
    float * __restrict__ partial_sum,
    const float * __restrict__ query,
    const float * __restrict__ k_cache,
    const float * __restrict__ v_cache,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int current_seq_len,
    int chunk_size,
    int num_chunks
);

extern __global__ void flash_decoding_combine_kernel(
    float * __restrict__ output,
    const float * __restrict__ partial_out,
    const float * __restrict__ partial_max,
    const float * __restrict__ partial_sum,
    int head_dim,
    int num_chunks
);

extern "C" {
void launch_flash_decoding(
    float *output,
    float *partial_out,
    float *partial_max,
    float *partial_sum,
    const float *query,
    const float *k_cache,
    const float *v_cache,
    int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int current_seq_len,
    const int chunk_size,
    void *stream_ptr
) {
    if (current_seq_len == 0) return;

    const auto stream = static_cast<cudaStream_t>(stream_ptr);
    const int num_chunks = (current_seq_len + chunk_size - 1) / chunk_size;

    constexpr int threads_p1 = 128;
    const size_t shared_mem_size = (head_dim + chunk_size) * sizeof(float);
    dim3 blocks_p1(num_heads, num_chunks);

    flash_decoding_partial_kernel<<<blocks_p1, threads_p1, shared_mem_size, stream>>>(
        partial_out, partial_max, partial_sum,
        query, k_cache, v_cache,
        num_heads, num_kv_heads, head_dim,
        current_seq_len, chunk_size, num_chunks
    );

    constexpr int threads_p2 = 32;
    flash_decoding_combine_kernel<<<num_heads, threads_p2, 0, stream>>>(
        output, partial_out, partial_max, partial_sum,
        head_dim, num_chunks
    );
}
}
