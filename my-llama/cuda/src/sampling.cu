#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

__device__ __forceinline__ float warp_reduce_max_sample(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum_sample(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void fused_sampling_kernel_v2(
    int * __restrict__ token_id,
    const float * __restrict__ logits,
    const float rand_val,
    const float temperature,
    const float top_p,
    const int vocab_size
) {
    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int num_warps = blockDim.x / 32;

    const float inv_temp = 1.0f / (temperature + 1e-9f);

    float max_val = -1e20f;
    int max_idx = 0;

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        const float lgt = logits[i] * inv_temp;
        if (lgt > max_val) {
            max_val = lgt;
            max_idx = i;
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        constexpr unsigned int mask = 0xFFFFFFFF;
        const float remote_max = __shfl_xor_sync(mask, max_val, offset);
        const int remote_idx = __shfl_xor_sync(mask, max_idx, offset);
        if (remote_max > max_val) {
            max_val = remote_max;
            max_idx = remote_idx;
        }
    }

    __shared__ float s_warp_maxes[32];
    __shared__ int s_warp_indices[32];
    __shared__ float s_block_max;
    __shared__ int s_block_max_idx;
    __shared__ float s_block_sum;

    if (lane_id == 0) {
        s_warp_maxes[warp_id] = max_val;
        s_warp_indices[warp_id] = max_idx;
    }
    __syncthreads();

    if (tid == 0) {
        float block_max = -1e20f;
        int block_max_idx = 0;

        for (int w = 0; w < num_warps; ++w) {
            if (s_warp_maxes[w] > block_max) {
                block_max = s_warp_maxes[w];
                block_max_idx = s_warp_indices[w];
            }
        }
        s_block_max = block_max;
        s_block_max_idx = block_max_idx;
    }
    __syncthreads();

    const float block_max = s_block_max;
    const int block_max_idx = s_block_max_idx;

    if (rand_val < 0.10f || temperature < 0.05f) {
        if (tid == 0) *token_id = block_max_idx;
        return;
    }

    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        const float exp_val = expf(logits[i] * inv_temp - block_max);
        local_sum += exp_val;
    }

    float block_sum = warp_reduce_sum_sample(local_sum);
    if (lane_id == 0) s_warp_maxes[warp_id] = block_sum;
    __syncthreads();

    if (warp_id == 0) {
        const float val = tid < num_warps ? s_warp_maxes[tid] : 0.0f;
        block_sum = warp_reduce_sum_sample(val);
        if (tid == 0) s_block_sum = block_sum;
    }
    __syncthreads();

    if (tid == 0) {
        const float inv_sum = 1.0f / (s_block_sum + 1e-9f);
        float cumulative_p = 0.0f;
        int selected = block_max_idx;

        for (int i = 0; i < vocab_size; ++i) {
            const float p = expf(logits[i] * inv_temp - block_max) * inv_sum;
            cumulative_p += p;
            if (rand_val <= cumulative_p || cumulative_p >= top_p) {
                selected = i;
                break;
            }
        }
        *token_id = selected;
    }
}

extern "C" {
void launch_fused_sampling(
    int *token_id,
    const float *logits,
    const float rand_val,
    const float temperature,
    const float top_p,
    const int vocab_size,
    void *stream_ptr
) {
    if (vocab_size == 0) return;

    constexpr int threads = 256;
    const auto stream = static_cast<cudaStream_t>(stream_ptr);

    fused_sampling_kernel_v2<<<1, threads, 0, stream>>>(
        token_id, logits, rand_val, temperature, top_p, vocab_size
    );
}
}
