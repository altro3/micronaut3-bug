#include <cuda_runtime.h>

__global__ void attention_scores_kernel(float *const scores,
                                        const float *const query,
                                        const float *const k_cache,
                                        const int num_heads,
                                        const int num_kv_heads,
                                        const int head_dim,
                                        const int current_seq_len) {
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
    s_dot[tid] = local_sum;
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

__global__ void softmax_attention_kernel(float *const scores, const int current_seq_len) {
    const int head_idx = blockIdx.x;
    float *const row = scores + head_idx * current_seq_len;

    const int tid = threadIdx.x;

    extern __shared__ float s_mem[];
    float local_max = -INFINITY;

    for (int i = tid; i < current_seq_len; i += blockDim.x) {
        if (row[i] > local_max) {
            local_max = row[i];
        }
    }
    s_mem[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_mem[tid + s] > s_mem[tid]) {
                s_mem[tid] = s_mem[tid + s];
            }
        }
        __syncthreads();
    }
    const float row_max = s_mem[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < current_seq_len; i += blockDim.x) {
        local_sum += expf(row[i] - row_max);
    }
    s_mem[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_mem[tid] += s_mem[tid + s];
        }
        __syncthreads();
    }
    const float sum_total = s_mem[0];

    for (int i = tid; i < current_seq_len; i += blockDim.x) {
        row[i] = expf(row[i] - row_max) / sum_total;
    }
}

__global__ void attention_values_kernel(float *const output,
                                        const float *const probabilities,
                                        const float *const v_cache,
                                        const int num_heads,
                                        const int num_kv_heads,
                                        const int head_dim,
                                        const int current_seq_len) {
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (tid < head_dim) {
        const int kv_head_ratio = num_heads / num_kv_heads;
        const int kv_head_idx = head_idx / kv_head_ratio;
        const int hidden_size = num_heads * head_dim;

        float sum = 0.0f;

        for (int tok = 0; tok < current_seq_len; ++tok) {
            const float prob = probabilities[head_idx * current_seq_len + tok];
            const int v_offset = tok * hidden_size + kv_head_idx * head_dim + tid;
            const float v_val = v_cache[v_offset];

            sum += prob * v_val;
        }

        output[head_idx * head_dim + tid] = sum;
    }
}


extern "C" {
void launch_attention_scores(float *output_scores,
                             const float *query,
                             const float *k_cache,
                             const int num_heads,
                             const int num_kv_heads,
                             const int head_dim,
                             const int current_seq_len) {
    const int threads = head_dim < 256 ? head_dim : 256;
    const int shared_mem_size = threads * sizeof(float);
    dim3 blocks_per_grid(current_seq_len, num_heads);

    attention_scores_kernel<<<blocks_per_grid, threads, shared_mem_size>>>(
        output_scores, query, k_cache, num_heads, num_kv_heads, head_dim, current_seq_len
    );
}

void launch_softmax_attention(float *scores, int num_heads, const int current_seq_len) {
    const int threads = current_seq_len < 256 ? current_seq_len : 256;
    const int shared_mem_size = threads * sizeof(float);

    softmax_attention_kernel<<<num_heads, threads, shared_mem_size>>>(scores, current_seq_len);
}

void launch_attention_values(float *output,
                             const float *probabilities,
                             const float *v_cache,
                             int num_heads,
                             const int num_kv_heads,
                             const int head_dim,
                             const int current_seq_len) {
    const int threads = head_dim < 256 ? head_dim : 256;

    attention_values_kernel<<<num_heads, threads>>>(
        output, probabilities, v_cache, num_heads, num_kv_heads, head_dim, current_seq_len
    );
}
}
