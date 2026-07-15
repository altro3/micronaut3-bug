#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void rope_forward_kernel_ultimate(
    float * __restrict__ vec,
    const int * __restrict__ positions,
    const float * __restrict__ inv_freq,
    const int num_heads,
    const int head_dim,
    const int total_tokens
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;

    const int pos = positions[token_idx];
    const float pos_f = static_cast<float>(pos);
    const int tid = threadIdx.x;

    const int half_dim = head_dim / 2;
    const int half_dim_f4 = half_dim / 4;
    const int total_valid_elements_f4 = num_heads * half_dim_f4;

    for (int global_f4_idx = tid; global_f4_idx < total_valid_elements_f4; global_f4_idx += blockDim.x) {
        const int head_idx = global_f4_idx / half_dim_f4;
        const int feature_idx_f4 = global_f4_idx % half_dim_f4;

        const long long base_offset_v0 = (static_cast<long long>(token_idx) * num_heads + head_idx) * head_dim + feature_idx_f4 * 4;
        const long long base_offset_v1 = base_offset_v0 + half_dim;

        const float4 v0 = __ldcs(reinterpret_cast<const float4 *>(&vec[base_offset_v0]));
        const float4 v1 = __ldcs(reinterpret_cast<const float4 *>(&vec[base_offset_v1]));
        const float4 freq = __ldcs(reinterpret_cast<const float4 *>(&inv_freq[feature_idx_f4 * 4]));

        float sin_0, cos_0, sin_1, cos_1, sin_2, cos_2, sin_3, cos_3;
        __sincosf(pos_f * freq.x, &sin_0, &cos_0);
        __sincosf(pos_f * freq.y, &sin_1, &cos_1);
        __sincosf(pos_f * freq.z, &sin_2, &cos_2);
        __sincosf(pos_f * freq.w, &sin_3, &cos_3);

        float4 out_v0, out_v1;
        out_v0.x = v0.x * cos_0 - v1.x * sin_0;
        out_v1.x = v0.x * sin_0 + v1.x * cos_0;

        out_v0.y = v0.y * cos_1 - v1.y * sin_1;
        out_v1.y = v0.y * sin_1 + v1.y * cos_1;

        out_v0.z = v0.z * cos_2 - v1.z * sin_2;
        out_v1.z = v0.z * sin_2 + v1.z * cos_2;

        out_v0.w = v0.w * cos_3 - v1.w * sin_3;
        out_v1.w = v0.w * sin_3 + v1.w * cos_3;

        __stcs(reinterpret_cast<float4 *>(&vec[base_offset_v0]), out_v0);
        __stcs(reinterpret_cast<float4 *>(&vec[base_offset_v1]), out_v1);
    }
}

__global__ void rope_backward_kernel_ultimate(
    float * __restrict__ grad_in,
    const int * __restrict__ positions,
    const float * __restrict__ inv_freq,
    const int num_heads,
    const int head_dim,
    const int total_tokens
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;

    const int pos = positions[token_idx];
    const float pos_f = static_cast<float>(pos);
    const int tid = threadIdx.x;

    const int half_dim = head_dim / 2;
    const int half_dim_f4 = half_dim / 4;
    const int total_valid_elements_f4 = num_heads * half_dim_f4;

    for (int global_f4_idx = tid; global_f4_idx < total_valid_elements_f4; global_f4_idx += blockDim.x) {
        const int head_idx = global_f4_idx / half_dim_f4;
        const int feature_idx_f4 = global_f4_idx % half_dim_f4;

        const long long base_offset_g0 = (static_cast<long long>(token_idx) * num_heads + head_idx) * head_dim + feature_idx_f4 * 4;
        const long long base_offset_g1 = base_offset_g0 + half_dim;

        const float4 g0 = __ldcs(reinterpret_cast<const float4 *>(&grad_in[base_offset_g0]));
        const float4 g1 = __ldcs(reinterpret_cast<const float4 *>(&grad_in[base_offset_g1]));
        const float4 freq = __ldcs(reinterpret_cast<const float4 *>(&inv_freq[feature_idx_f4 * 4]));

        float sin_0, cos_0, sin_1, cos_1, sin_2, cos_2, sin_3, cos_3;
        __sincosf(pos_f * freq.x, &sin_0, &cos_0);
        __sincosf(pos_f * freq.y, &sin_1, &cos_1);
        __sincosf(pos_f * freq.z, &sin_2, &cos_2);
        __sincosf(pos_f * freq.w, &sin_3, &cos_3);

        float4 out_g0, out_g1;
        out_g0.x = g0.x * cos_0 + g1.x * sin_0;
        out_g1.x = -g0.x * sin_0 + g1.x * cos_0;

        out_g0.y = g0.y * cos_1 + g1.y * sin_1;
        out_g1.y = -g0.y * sin_1 + g1.y * cos_1;

        out_g0.z = g0.z * cos_2 + g1.z * sin_2;
        out_g1.z = -g0.z * sin_2 + g1.z * cos_2;

        out_g0.w = g0.w * cos_3 + g1.w * sin_3;
        out_g1.w = -g0.w * sin_3 + g1.w * cos_3;

        __stcs(reinterpret_cast<float4 *>(&grad_in[base_offset_g0]), out_g0);
        __stcs(reinterpret_cast<float4 *>(&grad_in[base_offset_g1]), out_g1);
    }
}

extern "C" {
void launch_rope_forward(
    float *vec,
    const int *positions,
    const float *inv_freq,
    const int num_heads,
    const int head_dim,
    const int total_tokens,
    void *stream_ptr
) {
    if (total_tokens == 0 || num_heads == 0 || head_dim == 0) return;

    constexpr int threads = 256;
    const int blocks = total_tokens;
    const auto stream = static_cast<cudaStream_t>(stream_ptr);

    rope_forward_kernel_ultimate<<<blocks, threads, 0, stream>>>(
        vec, positions, inv_freq, num_heads, head_dim, total_tokens
    );
}

void launch_rope_backward(
    float *grad_in,
    const int *positions,
    const float *inv_freq,
    const int num_heads,
    const int head_dim,
    const int total_tokens,
    void *stream_ptr
) {
    if (total_tokens == 0 || num_heads == 0 || head_dim == 0) return;

    constexpr int threads = 256;
    const int blocks = total_tokens;
    const auto stream = static_cast<cudaStream_t>(stream_ptr);

    rope_backward_kernel_ultimate<<<blocks, threads, 0, stream>>>(
        grad_in, positions, inv_freq, num_heads, head_dim, total_tokens
    );
}
}
