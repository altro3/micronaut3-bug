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

    const unsigned int token_id = tokens[token_idx];
    const int tid = threadIdx.x;

    float *const out_ptr = out + static_cast<long long>(token_idx) * out_features;
    const long long weight_row_offset = static_cast<long long>(token_id) * out_features;

    const int out_features_f16 = out_features / 16;
    const int stride = blockDim.x;

    if (token_id < vocab_size) {
        for (int idx_f16 = tid; idx_f16 < out_features_f16; idx_f16 += stride) {
            const long long offset = weight_row_offset + idx_f16 * 16;
            const long long out_offset = idx_f16 * 16;

            const float4 w0 = __ldcs(reinterpret_cast<const float4 *>(&weight[offset]));
            const float4 w1 = __ldcs(reinterpret_cast<const float4 *>(&weight[offset + 4]));
            const float4 w2 = __ldcs(reinterpret_cast<const float4 *>(&weight[offset + 8]));
            const float4 w3 = __ldcs(reinterpret_cast<const float4 *>(&weight[offset + 12]));

            __stcs(reinterpret_cast<float4 *>(&out_ptr[out_offset]), w0);
            __stcs(reinterpret_cast<float4 *>(&out_ptr[out_offset + 4]), w1);
            __stcs(reinterpret_cast<float4 *>(&out_ptr[out_offset + 8]), w2);
            __stcs(reinterpret_cast<float4 *>(&out_ptr[out_offset + 12]), w3);
        }
    } else {
        const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int idx_f16 = tid; idx_f16 < out_features_f16; idx_f16 += stride) {
            const long long out_offset = idx_f16 * 16;

            __stcs(reinterpret_cast<float4 *>(&out_ptr[out_offset]), zero);
            __stcs(reinterpret_cast<float4 *>(&out_ptr[out_offset + 4]), zero);
            __stcs(reinterpret_cast<float4 *>(&out_ptr[out_offset + 8]), zero);
            __stcs(reinterpret_cast<float4 *>(&out_ptr[out_offset + 12]), zero);
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
    void *stream_ptr
) {
    if (total_tokens == 0 || out_features == 0) return;

    constexpr int threads = 256;
    const int blocks = total_tokens;
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
