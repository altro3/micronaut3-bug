#include <cuda_runtime.h>

__global__ void attention_values_kernel(
    float * __restrict__ output,
    const float * __restrict__ probabilities,
    const float * __restrict__ v_cache,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int current_seq_len
) {
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
void launch_attention_values(float *output,
                             const float *probabilities,
                             const float *v_cache,
                             int num_heads,
                             const int num_kv_heads,
                             const int head_dim,
                             const int current_seq_len,
                             void *stream_ptr) {
    const int threads = head_dim < 256 ? (head_dim + 31) / 32 * 32 : 256;

    const auto stream = static_cast<cudaStream_t>(stream_ptr);

    attention_values_kernel<<<num_heads, threads, 0, stream>>>(
        output, probabilities, v_cache, num_heads, num_kv_heads, head_dim, current_seq_len
    );
}
}
