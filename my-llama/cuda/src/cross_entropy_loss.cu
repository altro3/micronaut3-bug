#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

__device__ __forceinline__ float warp_reduce_max_loss(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum_loss(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void fused_cross_entropy_kernel(
    float * __restrict__ logits,
    const int * __restrict__ targets,
    float * __restrict__ losses,
    const int total_tokens,
    const int vocab_size
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;

    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int num_warps = blockDim.x / 32;

    const int target_label = targets[token_idx];
    float *const token_logits = logits + static_cast<long long>(token_idx) * vocab_size;

    __shared__ float s_warp_max[32];
    __shared__ float s_warp_sum[32];

    float local_max = -1e20f;
    for (int v = tid; v < vocab_size; v += blockDim.x) {
        local_max = fmaxf(local_max, token_logits[v]);
    }

    float block_max = warp_reduce_max_loss(local_max);
    if (lane_id == 0) s_warp_max[warp_id] = block_max;
    __syncthreads();

    if (warp_id == 0) {
        const float val = tid < num_warps ? s_warp_max[lane_id] : -1e20f;
        block_max = warp_reduce_max_loss(val);
        s_warp_max[0] = __shfl_sync(0xFFFFFFFF, block_max, 0);
    }
    __syncthreads();
    block_max = s_warp_max[0];

    float local_sum = 0.0f;
    for (int v = tid; v < vocab_size; v += blockDim.x) {
        local_sum += expf(token_logits[v] - block_max);
    }

    float block_sum = warp_reduce_sum_loss(local_sum);
    if (lane_id == 0) s_warp_sum[warp_id] = block_sum;
    __syncthreads();

    if (warp_id == 0) {
        const float val = tid < num_warps ? s_warp_sum[lane_id] : 0.0f;
        block_sum = warp_reduce_sum_loss(val);
        s_warp_sum[0] = __shfl_sync(0xFFFFFFFF, block_sum, 0);
    }
    __syncthreads();
    block_sum = s_warp_sum[0];

    const float target_logit = (target_label >= 0 && target_label < vocab_size) ? token_logits[target_label] : 0.0f;
    if (tid == 0) {
        losses[token_idx] = logf(block_sum) + block_max - target_logit;
    }

    const float inv_block_sum = 1.0f / (block_sum + 1e-9f);
    const int vocab_size_f4 = vocab_size / 4;

    for (int v_f4 = tid; v_f4 < vocab_size_f4; v_f4 += blockDim.x) {
        const int base_v = v_f4 * 4;
        float4 logit_val = *reinterpret_cast<const float4 *>(&token_logits[base_v]);

        logit_val.x = expf(logit_val.x - block_max) * inv_block_sum;
        logit_val.y = expf(logit_val.y - block_max) * inv_block_sum;
        logit_val.z = expf(logit_val.z - block_max) * inv_block_sum;
        logit_val.w = expf(logit_val.w - block_max) * inv_block_sum;

        if (target_label >= base_v && target_label < base_v + 4) {
            const int offset = target_label - base_v;
            if (offset == 0) logit_val.x -= 1.0f;
            else if (offset == 1) logit_val.y -= 1.0f;
            else if (offset == 2) logit_val.z -= 1.0f;
            else if (offset == 3) logit_val.w -= 1.0f;
        }

        *reinterpret_cast<float4 *>(&token_logits[base_v]) = logit_val;
    }
}

extern "C" {
void launch_cross_entropy_loss(
    float *logits,
    const int *targets,
    float *losses,
    const int total_tokens,
    const int vocab_size,
    void *stream_ptr
) {
    if (total_tokens == 0 || vocab_size == 0) return;

    constexpr int threads = 256;
    const int blocks = total_tokens;
    const auto stream = static_cast<cudaStream_t>(stream_ptr);

    fused_cross_entropy_kernel<<<blocks, threads, 0, stream>>>(
        logits, targets, losses, total_tokens, vocab_size
    );
}
}
