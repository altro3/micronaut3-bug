#include <cuda_runtime.h>

__global__ void embeddings_kernel(
    float *out,
    const float *weight,
    const unsigned int *tokens,
    const int total_tokens,
    const int out_features,
    const int vocab_size
) {
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = total_tokens * out_features;

    if (global_idx < total_elements) {
        const int token_idx = global_idx / out_features;
        const int feature_idx = global_idx % out_features;

        const unsigned int token_id = tokens[token_idx];

        if (token_id < vocab_size) {
            const long long weight_idx = static_cast<long long>(token_id) * out_features + feature_idx;
            out[global_idx] = weight[weight_idx];
        } else {
            out[global_idx] = 0.0f;
        }
    }
}

extern "C" {
void launch_embeddings(
    float *out,
    const float *weight,
    const unsigned int *tokens,
    const int total_tokens,
    const int out_features,
    const int vocab_size,
    cudaStream_t stream
) {
    const int total_elements = total_tokens * out_features;
    if (total_elements == 0) return;

    int threads_per_block = 256;
    int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    embeddings_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        out,
        weight,
        tokens,
        total_tokens,
        out_features,
        vocab_size
    );
}
}
