#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

__device__ __forceinline__ float warp_reduce_max_attn(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum_attn(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void fused_attention_kernel(
    float * __restrict__ output,
    const float * __restrict__ query,
    const float * __restrict__ k_cache,
    const float * __restrict__ v_cache,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int current_seq_len
) {
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int num_warps = blockDim.x / 32;

    const int kv_head_ratio = num_heads / num_kv_heads;
    const int kv_head_idx = head_idx / kv_head_ratio;

    const float *const q_vec = query + head_idx * head_dim;
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    const int head_dim_f4 = head_dim / 4;

    __shared__ float s_warp_max[32];
    __shared__ float s_warp_sum[32];
    extern __shared__ float s_scores[];

    float local_max = -1e20f;
    for (int tok = tid; tok < current_seq_len; tok += blockDim.x) {
        const float *const k_vec = k_cache + (tok * num_kv_heads + kv_head_idx) * head_dim;
        float score_acc = 0.0f;

        for (int i = 0; i < head_dim_f4; ++i) {
            const float4 q_val = *reinterpret_cast<const float4 *>(&q_vec[i * 4]);
            const float4 k_val = *reinterpret_cast<const float4 *>(&k_vec[i * 4]);
            score_acc += q_val.x * k_val.x + q_val.y * k_val.y + q_val.z * k_val.z + q_val.w * k_val.w;
        }

        const float raw_score = score_acc * scale;
        s_scores[tok] = raw_score;
        local_max = fmaxf(local_max, raw_score);
    }

    float block_max = warp_reduce_max_attn(local_max);
    if (lane_id == 0) s_warp_max[warp_id] = block_max;
    __syncthreads();

    if (warp_id == 0) {
        const float val = tid < num_warps ? s_warp_max[lane_id] : -1e20f;
        block_max = warp_reduce_max_attn(val);
        s_warp_max[0] = __shfl_sync(0xFFFFFFFF, block_max, 0);
    }
    __syncthreads();
    block_max = s_warp_max[0];

    float local_sum = 0.0f;
    for (int tok = tid; tok < current_seq_len; tok += blockDim.x) {
        const float exp_score = expf(s_scores[tok] - block_max);
        s_scores[tok] = exp_score;
        local_sum += exp_score;
    }

    float block_sum = warp_reduce_sum_attn(local_sum);
    if (lane_id == 0) s_warp_sum[warp_id] = block_sum;
    __syncthreads();

    if (warp_id == 0) {
        const float val = tid < num_warps ? s_warp_sum[lane_id] : 0.0f;
        block_sum = warp_reduce_sum_attn(val);
        s_warp_sum[0] = __shfl_sync(0xFFFFFFFF, block_sum, 0);
    }
    __syncthreads();
    block_sum = s_warp_sum[0];

    const float inv_block_sum = 1.0f / (block_sum + 1e-9f);
    for (int tok = tid; tok < current_seq_len; tok += blockDim.x) {
        s_scores[tok] *= inv_block_sum;
    }
    __syncthreads();

    float *const out_ptr = output + head_idx * head_dim;
    for (int d = tid; d < head_dim_f4; d += blockDim.x) {
        float4 v_acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        for (int tok = 0; tok < current_seq_len; ++tok) {
            const float prob = s_scores[tok];
            const float *const v_row = v_cache + (tok * num_kv_heads + kv_head_idx) * head_dim;
            const float4 v_val = __ldcs(reinterpret_cast<const float4 *>(&v_row[d * 4]));

            v_acc.x += prob * v_val.x;
            v_acc.y += prob * v_val.y;
            v_acc.z += prob * v_val.z;
            v_acc.w += prob * v_val.w;
        }

        *reinterpret_cast<float4 *>(&out_ptr[d * 4]) = v_acc;
    }
}

extern "C" {
void launch_fused_attention(
    float *output,
    const float *query,
    const float *k_cache,
    const float *v_cache,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int current_seq_len,
    void *stream_ptr
) {
    constexpr int threads = 128;
    const size_t shared_mem_size = current_seq_len * sizeof(float);
    const auto stream = static_cast<cudaStream_t>(stream_ptr);

    fused_attention_kernel<<<num_heads, threads, shared_mem_size, stream>>>(
        output, query, k_cache, v_cache, num_heads, num_kv_heads, head_dim, current_seq_len
    );
}
}
