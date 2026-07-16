#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp8.h>
#include <stdint.h>

#define BLOCK_M 64
#define BLOCK_N 64
#define PADDING 16

template<int HEAD_DIM>
__global__ void flash_attention_fp8_blackwell_prefill_kernel(
    __nv_fp8_e4m3 * __restrict__ output,
    const __nv_fp8_e4m3 * __restrict__ query,
    const __nv_fp8_e4m3 * __restrict__ key,
    const __nv_fp8_e4m3 * __restrict__ value,
    int q_seq_len,
    int kv_seq_len,
    int num_heads,
    int num_kv_heads,
    float scale,
    int batch_size
);

template<int HEAD_DIM>
constexpr size_t get_prefill_shared_mem_size() {
    const int stride_q = HEAD_DIM + PADDING;
    const int stride_k = HEAD_DIM + PADDING;
    constexpr int stride_v = BLOCK_N + PADDING;

    const size_t s_q_size = BLOCK_M * stride_q + 15 & ~15;
    const size_t s_k_size = BLOCK_N * stride_k + 15 & ~15;
    const size_t s_v_size = HEAD_DIM * stride_v + 15 & ~15;

    constexpr size_t s_stats_size = BLOCK_M * sizeof(float) + BLOCK_M * sizeof(float);
    const size_t s_acc_size = BLOCK_M * HEAD_DIM * sizeof(int) + 15 & ~15;
    constexpr size_t s_scores_size = BLOCK_M * BLOCK_N * sizeof(int) + 15 & ~15;
    constexpr size_t s_scores_u8_size = BLOCK_M * BLOCK_N;

    return s_q_size + s_k_size + s_v_size + s_stats_size + s_acc_size + s_scores_size + s_scores_u8_size;
}

extern "C" {
void init_flash_attention_prefill_kernels(size_t max_shared_mem) {
    cudaFuncSetAttribute(flash_attention_fp8_blackwell_prefill_kernel<256>, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);
    cudaFuncSetAttribute(flash_attention_fp8_blackwell_prefill_kernel<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);
    cudaFuncSetAttribute(flash_attention_fp8_blackwell_prefill_kernel<64>, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);
}

void launch_flash_attention_prefill_fp8(
    __nv_fp8_e4m3 *output,
    const __nv_fp8_e4m3 *query,
    const __nv_fp8_e4m3 *key,
    const __nv_fp8_e4m3 *value,
    const int batch_size,
    const int q_seq_len,
    const int kv_seq_len,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int threads_per_block,
    const float scale,
    cudaStream_t stream
) {
    if (q_seq_len == 0 || kv_seq_len == 0) return;

    const int num_m_tiles = (q_seq_len + BLOCK_M - 1) / BLOCK_M;
    dim3 grid(num_heads * batch_size, num_m_tiles);

    if (head_dim == 256) {
        static constexpr size_t shmem_256 = get_prefill_shared_mem_size<256>();
        flash_attention_fp8_blackwell_prefill_kernel<256><<<grid, threads_per_block, shmem_256, stream>>>(output, query, key, value, q_seq_len, kv_seq_len, num_heads, num_kv_heads, scale, batch_size);
    } else if (head_dim == 128) {
        static constexpr size_t shmem_128 = get_prefill_shared_mem_size<128>();
        flash_attention_fp8_blackwell_prefill_kernel<128><<<grid, threads_per_block, shmem_128, stream>>>(output, query, key, value, q_seq_len, kv_seq_len, num_heads, num_kv_heads, scale, batch_size);
    } else if (head_dim == 64) {
        static constexpr size_t shmem_64 = get_prefill_shared_mem_size<64>();
        flash_attention_fp8_blackwell_prefill_kernel<64><<<grid, threads_per_block, shmem_64, stream>>>(output, query, key, value, q_seq_len, kv_seq_len, num_heads, num_kv_heads, scale, batch_size);
    }
}
}
