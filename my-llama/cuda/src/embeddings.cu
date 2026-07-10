#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void embeddings_kernel(
    float * __restrict__ out,
    const float * __restrict__ weight,
    const unsigned int * __restrict__ tokens,
    const int total_tokens,
    const int out_features,
    const int vocab_size
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;

    __shared__ unsigned int s_token_id;
    if (threadIdx.x == 0) {
        s_token_id = tokens[token_idx];
    }
    __syncthreads();

    const unsigned int token_id = s_token_id;
    const int out_features_f8 = out_features / 8;
    const int feature_idx_f8 = blockIdx.y * blockDim.x + threadIdx.x;

    if (feature_idx_f8 >= out_features_f8) return;

    float *const out_ptr = out + static_cast<long long>(token_idx) * out_features + feature_idx_f8 * 8;

    if (token_id < vocab_size) {
        const long long weight_idx = static_cast<long long>(token_id) * out_features + feature_idx_f8 * 8;

        float4 w0 = *reinterpret_cast<const float4 *>(&weight[weight_idx]);
        float4 w1 = *reinterpret_cast<const float4 *>(&weight[weight_idx + 4]);

        *reinterpret_cast<float4 *>(out_ptr) = w0;
        *reinterpret_cast<float4 *>(out_ptr + 4) = w1;
    } else {
        float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        *reinterpret_cast<float4 *>(out_ptr) = zero;
        *reinterpret_cast<float4 *>(out_ptr + 4) = zero;
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
    void *stream_ptr
) {
    if (total_tokens == 0 || out_features == 0) return;

    const int out_features_f8 = out_features / 8;
    constexpr int threads = 256;

    dim3 blocks(total_tokens, (out_features_f8 + threads - 1) / threads);
    const auto stream = static_cast<cudaStream_t>(stream_ptr);

    embeddings_kernel<<<blocks, threads, 0, stream>>>(
        out,
        weight,
        tokens,
        total_tokens,
        out_features,
        vocab_size
    );
}
}
