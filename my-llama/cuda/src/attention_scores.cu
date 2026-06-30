#include <cuda_runtime.h>

__global__ void attention_scores_kernel(
    float * __restrict__ scores,
    const float * __restrict__ query,
    const float * __restrict__ k_cache,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int current_seq_len
) {
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;

    if (token_idx >= current_seq_len) return;

    const int tid = threadIdx.x;

    const int kv_head_ratio = num_heads / num_kv_heads;
    const int kv_head_idx = head_idx / kv_head_ratio;

    const float *const q_vec = query + head_idx * head_dim;
    const float *const k_vec = k_cache + (token_idx * num_kv_heads + kv_head_idx) * head_dim;

    extern __shared__ float s_dot[];
    float local_sum = 0.0f;

    for (int i = tid; i < head_dim; i += blockDim.x) {
        local_sum += q_vec[i] * k_vec[i];
    }

    s_dot[tid] = tid < head_dim ? local_sum : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_dot[tid] += s_dot[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
        scores[head_idx * current_seq_len + token_idx] = s_dot[0] * scale;
    }
}

extern "C" {
void launch_attention_scores(float *output_scores,
                             const float *query,
                             const float *k_cache,
                             const int num_heads,
                             const int num_kv_heads,
                             const int head_dim,
                             const int current_seq_len,
                             void *stream_ptr) {
    const int threads = head_dim < 256 ? (head_dim + 31) / 32 * 32 : 256;
    const size_t shared_mem_size = threads * sizeof(float);
    dim3 blocks_per_grid(current_seq_len, num_heads);

    const auto stream = static_cast<cudaStream_t>(stream_ptr);

    attention_scores_kernel<<<blocks_per_grid, threads, shared_mem_size, stream>>>(
        output_scores, query, k_cache, num_heads, num_kv_heads, head_dim, current_seq_len
    );
}
}
