#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdint.h>
#include "speculative_types.h"
#include "speculative_sampling.cuh"

__device__ __forceinline__ float warp_reduce_max_spec(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum_spec(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void speculative_verify_kernel(
    int32_t * __restrict__ accepted_tokens,
    int32_t * __restrict__ num_accepted,
    const float * __restrict__ target_logits,
    const float * __restrict__ draft_probs,
    const int32_t * __restrict__ draft_tokens,
    const float * __restrict__ random_nums,
    float * __restrict__ workspace,
    const int vocab_size,
    const int max_draft_tokens,
    const float temperature
) {
    const int seq_idx = blockIdx.y;
    const int tid = threadIdx.x;

    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int num_warps = blockDim.x / 32;

    __shared__ SpecSharedState s_state;
    __shared__ int32_t s_sampled_token;
    __shared__ float s_warp_totals[32];

    if (tid == 0 && blockIdx.x == 0) {
        num_accepted[seq_idx] = 0;
        s_state.rejected = false;
        s_state.accepted_count = 0;
    }
    __syncthreads();

    float *s_target_probs = workspace + static_cast<long long>(seq_idx) * vocab_size;
    const float inv_temp = 1.0f / (temperature + 1e-9f);

    for (int step = 0; step < max_draft_tokens; ++step) {
        if (s_state.rejected) {
            if (tid == 0) num_accepted[seq_idx] = s_state.accepted_count;
            return;
        }

        const long long logit_offset = (static_cast<long long>(seq_idx) * max_draft_tokens + step) * vocab_size;
        const float *const step_logits = target_logits + logit_offset;

        float local_max = -1e20f;
        for (int i = tid; i < vocab_size; i += blockDim.x) {
            local_max = fmaxf(local_max, step_logits[i]);
        }

        const float warp_max = warp_reduce_max_spec(local_max);
        if (lane_id == 0 && warp_id < 32) s_state.warp_exchange[warp_id] = warp_max;
        __syncthreads();

        if (warp_id == 0) {
            float block_max_val = tid < num_warps ? s_state.warp_exchange[tid] : -1e20f;
            block_max_val = warp_reduce_max_spec(block_max_val);
            if (tid == 0) s_state.max_logit = block_max_val;
        }
        __syncthreads();

        float local_sum = 0.0f;
        for (int i = tid; i < vocab_size; i += blockDim.x) {
            const float p = __expf((step_logits[i] - s_state.max_logit) * inv_temp);
            s_target_probs[i] = p;
            local_sum += p;
        }

        const float warp_sum = warp_reduce_sum_spec(local_sum);
        if (lane_id == 0 && warp_id < 32) s_state.warp_exchange[warp_id] = warp_sum;
        __syncthreads();

        if (warp_id == 0) {
            float block_sum_val = tid < num_warps ? s_state.warp_exchange[tid] : 0.0f;
            block_sum_val = warp_reduce_sum_spec(block_sum_val);
            if (tid == 0) s_state.total_sum = block_sum_val;
        }
        __syncthreads();

        const float inv_total_sum = 1.0f / (s_state.total_sum + 1e-9f);
        for (int i = tid; i < vocab_size; i += blockDim.x) {
            s_target_probs[i] *= inv_total_sum;
        }
        __syncthreads();

        if (tid == 0) {
            const int32_t drafted_tok = draft_tokens[seq_idx * max_draft_tokens + step];
            const float p_target = s_target_probs[drafted_tok];
            const float q_draft = draft_probs[seq_idx * max_draft_tokens + step];
            const float r = random_nums[seq_idx * max_draft_tokens + step];

            if (p_target >= q_draft || r < p_target / (q_draft + 1e-9f)) {
                accepted_tokens[seq_idx * (max_draft_tokens + 1) + s_state.accepted_count] = drafted_tok;
                s_state.accepted_count++;
            } else {
                s_state.rejected = true;
            }
        }
        __syncthreads();

        if (s_state.rejected) {
            const float q_draft = draft_probs[seq_idx * max_draft_tokens + step];

            float local_norm_sum = 0.0f;
            for (int i = tid; i < vocab_size; i += blockDim.x) {
                const float diff = fmaxf(0.0f, s_target_probs[i] - q_draft);
                s_target_probs[i] = diff;
                local_norm_sum += diff;
            }

            const float warp_norm_sum = warp_reduce_sum_spec(local_norm_sum);
            if (lane_id == 0 && warp_id < 32) s_state.warp_exchange[warp_id] = warp_norm_sum;
            __syncthreads();

            if (warp_id == 0) {
                float block_norm_val = tid < num_warps ? s_state.warp_exchange[tid] : 0.0f;
                block_norm_val = warp_reduce_sum_spec(block_norm_val);
                if (tid == 0) s_state.norm_sum = block_norm_val;
            }
            __syncthreads();

            if (s_state.norm_sum > 1e-6f) {
                const float inv_norm = 1.0f / s_state.norm_sum;
                for (int i = tid; i < vocab_size; i += blockDim.x) s_target_probs[i] *= inv_norm;
            } else {
                for (int i = tid; i < vocab_size; i += blockDim.x) s_target_probs[i] = inv_total_sum;
            }
            __syncthreads();

            const int32_t recovery_token = sample_token_parallel(
                s_target_probs, random_nums[seq_idx * max_draft_tokens + max_draft_tokens], vocab_size,
                tid, lane_id, warp_id, num_warps, s_warp_totals, &s_state, &s_sampled_token
            );

            if (tid == 0) {
                accepted_tokens[seq_idx * (max_draft_tokens + 1) + s_state.accepted_count] = recovery_token;
                num_accepted[seq_idx] = s_state.accepted_count;
            }
            return;
        }
    }

    if (!s_state.rejected) {
        const int32_t bonus_token = sample_token_parallel(
            s_target_probs, random_nums[seq_idx * max_draft_tokens + max_draft_tokens], vocab_size,
            tid, lane_id, warp_id, num_warps, s_warp_totals, &s_state, &s_sampled_token
        );

        if (tid == 0) {
            accepted_tokens[seq_idx * (max_draft_tokens + 1) + max_draft_tokens] = bonus_token;
            s_state.accepted_count++;
            num_accepted[seq_idx] = s_state.accepted_count;
        }
    }
}
