#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdint.h>

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
    const int num_warps = (blockDim.x + 31) / 32;

    const int kv_head_ratio = num_heads / num_kv_heads;
    const int kv_head_idx = head_idx / kv_head_ratio;

    const float *const q_vec = query + head_idx * head_dim;
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    const int head_dim_f4 = head_dim / 4;

    extern __shared__ uint8_t s_dynamic_mem[];

    const int num_warps_aligned = num_warps + 3 & ~3;
    const auto s_warp_max = reinterpret_cast<float *>(s_dynamic_mem);
    float *s_warp_sum = s_warp_max + num_warps_aligned;
    float *s_scores = s_warp_sum + num_warps_aligned;

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
        float max_val = -1e20f;
        for (int w = lane_id; w < num_warps; w += 32) {
            max_val = fmaxf(max_val, s_warp_max[w]);
        }

        block_max = warp_reduce_max_attn(max_val);
        if (lane_id == 0) s_warp_max[0] = block_max;
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
        float sum_val = 0.0f;
        for (int w = lane_id; w < num_warps; w += 32) {
            sum_val += s_warp_sum[w];
        }

        block_sum = warp_reduce_sum_attn(sum_val);
        if (lane_id == 0) s_warp_sum[0] = block_sum;
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
    const int threads_per_block,
    void *stream_ptr
) {
    if (threads_per_block <= 0) return;

    const auto stream = static_cast<cudaStream_t>(stream_ptr);
    const int num_warps = (threads_per_block + 31) / 32;
    const int num_warps_aligned = num_warps + 3 & ~3;

    const size_t max_size = num_warps_aligned * sizeof(float);
    const size_t sum_size = num_warps_aligned * sizeof(float);
    const size_t scores_size = current_seq_len * sizeof(float);

    const size_t shared_mem_size = max_size + sum_size + scores_size;

    fused_attention_kernel<<<num_heads, threads_per_block, shared_mem_size, stream>>>(
        output, query, k_cache, v_cache, num_heads, num_kv_heads, head_dim, current_seq_len
    );
}
}
