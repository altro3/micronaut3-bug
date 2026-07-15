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

__global__ void fused_sampling_kernel(
    int * __restrict__ token_id,
    float * __restrict__ logits,
    const float rand_val,
    const float temperature,
    const int top_k,
    const float top_p,
    const int vocab_size
) {
    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int num_warps = blockDim.x / 32;

    __shared__ float s_warp_max[32];
    __shared__ float s_warp_sum[32];

    float local_max = -1e20f;
    const float inv_temp = 1.0f / (temperature + 1e-9f);

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        const float logit = logits[i] * inv_temp;
        logits[i] = logit;
        local_max = fmaxf(local_max, logit);
    }

    float block_max = warp_reduce_max_sample(local_max);
    if (lane_id == 0) s_warp_max[warp_id] = block_max;
    __syncthreads();

    if (warp_id == 0) {
        const float val = tid < num_warps ? s_warp_max[tid] : -1e20f;
        block_max = warp_reduce_max_sample(val);
        if (tid == 0) s_warp_max[0] = block_max;
    }
    __syncthreads();
    block_max = s_warp_max[0];

    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        const float exp_val = expf(logits[i] - block_max);
        logits[i] = exp_val;
        local_sum += exp_val;
    }

    float block_sum = warp_reduce_sum_sample(local_sum);
    if (lane_id == 0) s_warp_sum[warp_id] = block_sum;
    __syncthreads();

    if (warp_id == 0) {
        const float val = tid < num_warps ? s_warp_sum[tid] : 0.0f;
        block_sum = warp_reduce_sum_sample(val);
        if (tid == 0) s_warp_sum[0] = block_sum;
    }
    __syncthreads();
    block_sum = s_warp_sum[0];

    const float inv_block_sum = 1.0f / (block_sum + 1e-9f);
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        logits[i] *= inv_block_sum;
    }
    __syncthreads();

    extern __shared__ int s_top_indices[];
    float *s_top_probs = reinterpret_cast<float *>(&s_top_indices[top_k]);

    if (tid < top_k) {
        s_top_probs[tid] = -1.0f;
        s_top_indices[tid] = -1;
    }
    __syncthreads();

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        const float p = logits[i];
        if (p > s_top_probs[top_k - 1]) {
            for (int k = 0; k < top_k; ++k) {
                if (p > s_top_probs[k]) {
                    for (int m = top_k - 1; m > k; --m) {
                        s_top_probs[m] = s_top_probs[m - 1];
                        s_top_indices[m] = s_top_indices[m - 1];
                    }
                    s_top_probs[k] = p;
                    s_top_indices[k] = i;
                    break;
                }
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        float p_sum = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            if (s_top_indices[k] == -1) break;
            p_sum += s_top_probs[k];
        }

        const float inv_p_sum = 1.0f / (p_sum + 1e-9f);
        float cumulative_p = 0.0f;
        int selected_token = s_top_indices[0];

        for (int k = 0; k < top_k; ++k) {
            if (s_top_indices[k] == -1) break;

            const float norm_p = s_top_probs[k] * inv_p_sum;
            cumulative_p += norm_p;

            if (rand_val <= cumulative_p || cumulative_p >= top_p) {
                selected_token = s_top_indices[k];
                break;
            }
        }

        if (selected_token == -1) selected_token = 0;
        *token_id = selected_token;
    }
}

extern "C" {
void launch_fused_sampling(
    int *token_id,
    float *logits,
    const float rand_val,
    const float temperature,
    const int top_k,
    const float top_p,
    const int vocab_size,
    void *stream_ptr
) {
    if (vocab_size == 0) return;

    constexpr int threads = 256;
    const auto stream = static_cast<cudaStream_t>(stream_ptr);

    const int active_top_k = top_k > 0 && top_k <= 64 ? top_k : 32;
    size_t shared_mem_size = active_top_k * (sizeof(int) + sizeof(float));

    fused_sampling_kernel<<<1, threads, shared_mem_size, stream>>>(
        token_id, logits, rand_val, temperature, active_top_k, top_p, vocab_size
    );
}
}
