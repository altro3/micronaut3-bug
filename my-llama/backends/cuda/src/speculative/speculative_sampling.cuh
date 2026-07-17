#ifndef SPECULATIVE_SAMPLING_CUH
#define SPECULATIVE_SAMPLING_CUH

#include <cuda_runtime.h>
#include <stdint.h>
#include "speculative_types.h"

__device__ __forceinline__ int32_t warp_reduce_min_int(int32_t val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        int32_t tmp = __shfl_down_sync(0xFFFFFFFF, val, offset);
        val = tmp < val ? tmp : val;
    }
    return val;
}

__device__ __forceinline__ int32_t sample_token_parallel(
    const float * __restrict__ probs,
    const float r_sample,
    const int vocab_size,
    const int tid,
    const int lane_id,
    const int warp_id,
    const int num_warps,
    float *s_warp_totals,
    SpecSharedState *s_state,
    int32_t *s_sampled_token
) {
    if (tid == 0) {
        *s_sampled_token = vocab_size - 1;
    }

    float local_thread_sum = 0.0f;
    int32_t thread_sampled_tok = -1;
    bool found = false;

    const int items_per_thread = (vocab_size + blockDim.x - 1) / blockDim.x;
    const int start_idx = tid * items_per_thread;
    int end_idx = start_idx + items_per_thread;
    if (end_idx > vocab_size) end_idx = vocab_size;

    for (int i = start_idx; i < end_idx; ++i) {
        local_thread_sum += probs[i];
    }

    float warp_total_sum = local_thread_sum;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_total_sum += __shfl_down_sync(0xFFFFFFFF, warp_total_sum, offset);
    }

    if (lane_id == 0 && warp_id < 32) {
        s_warp_totals[warp_id] = warp_total_sum;
    }
    __syncthreads();

    __shared__ float s_warp_offsets[32];
    if (warp_id == 0) {
        const float w_val = tid < num_warps ? s_warp_totals[tid] : 0.0f;
        float w_prefix = w_val;
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const float remote_w = __shfl_up_sync(0xFFFFFFFF, w_prefix, offset);
            if (lane_id >= offset) w_prefix += remote_w;
        }
        if (tid < num_warps) {
            s_warp_offsets[tid] = w_prefix - w_val;
        }
    }
    __syncthreads();

    const float global_base = s_warp_offsets[warp_id];

    float warp_prefix = local_thread_sum;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const float remote_val = __shfl_up_sync(0xFFFFFFFF, warp_prefix, offset);
        if (lane_id >= offset) warp_prefix += remote_val;
    }
    const float thread_base = global_base + (warp_prefix - local_thread_sum);

    float current_cdf = thread_base;
    for (int i = start_idx; i < end_idx; ++i) {
        current_cdf += probs[i];
        if (!found && r_sample <= current_cdf) {
            thread_sampled_tok = i;
            found = true;
            break;
        }
    }

    int32_t warp_min_tok = warp_reduce_min_int(found ? thread_sampled_tok : vocab_size - 1);
    if (lane_id == 0 && warp_id < 32) {
        s_state->warp_exchange[warp_id] = *reinterpret_cast<float *>(&warp_min_tok);
    }
    __syncthreads();

    if (warp_id == 0) {
        int32_t block_min_tok = tid < num_warps ? *reinterpret_cast<int32_t *>(&s_state->warp_exchange[tid]) : (vocab_size - 1);
        block_min_tok = warp_reduce_min_int(block_min_tok);
        if (tid == 0) *s_sampled_token = block_min_tok;
    }
    __syncthreads();

    return *s_sampled_token;
}

#endif
