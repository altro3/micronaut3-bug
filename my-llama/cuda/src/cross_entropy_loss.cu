#include <cuda_fp16.h>
#include <cuda_runtime.h>

struct __align__(8) SoftmaxPair {
    float max_val;
    float sum_exp;
};

__device__ __forceinline__ void online_softmax_update(SoftmaxPair &curr, const float val) {
    const float old_max = curr.max_val;
    if (val > curr.max_val) {
        curr.max_val = val;
        curr.sum_exp = curr.sum_exp * __expf(old_max - val) + 1.0f;
    } else {
        curr.sum_exp += __expf(val - curr.max_val);
    }
}

__device__ __forceinline__ void warp_reduce_online(SoftmaxPair &curr) {
    const unsigned int mask = __activemask();
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float r_max = __shfl_xor_sync(mask, curr.max_val, offset);
        const float r_sum = __shfl_xor_sync(mask, curr.sum_exp, offset);
        if (r_max > curr.max_val) {
            curr.sum_exp = curr.sum_exp * __expf(curr.max_val - r_max) + r_sum;
            curr.max_val = r_max;
        } else {
            curr.sum_exp += r_sum * __expf(r_max - curr.max_val);
        }
    }
}

__global__ void __launch_bounds__(1024, 2) fused_cross_entropy_online_kernel(
    const float * __restrict__ logits,
    float * __restrict__ grads,
    const int * __restrict__ targets,
    float * __restrict__ losses,
    const int total_tokens,
    const int vocab_size_v4,
    const int vocab_size
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;

    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int num_warps = blockDim.x / 32;

    const long long token_offset = static_cast<long long>(token_idx) * vocab_size;
    const auto token_logits_v4 = reinterpret_cast<const float4 *>(logits + token_offset);
    auto out_grads_v4 = reinterpret_cast<float4 *>(grads + token_offset);

    extern __shared__ char shared_mem[];
    const auto s_max_pool = reinterpret_cast<float *>(shared_mem);
    const auto s_sum_pool = s_max_pool + num_warps;

    __shared__ float s_target_logit;
    __shared__ float s_final_max;
    __shared__ float s_final_sum;

    const int target_label = targets[token_idx];

    if (tid == 0) {
        if (target_label >= 0 && target_label < vocab_size) {
            s_target_logit = logits[token_offset + target_label];
        } else {
            s_target_logit = 0.0f;
        }
    }
    __syncthreads();

    SoftmaxPair local = {-1e20f, 0.0f};

    for (int v_f4 = tid; v_f4 < vocab_size_v4; v_f4 += blockDim.x) {
        const float4 log_v4 = __ldcs(&token_logits_v4[v_f4]);
        online_softmax_update(local, log_v4.x);
        online_softmax_update(local, log_v4.y);
        online_softmax_update(local, log_v4.z);
        online_softmax_update(local, log_v4.w);
    }

    warp_reduce_online(local);

    if (lane_id == 0) {
        s_max_pool[warp_id] = local.max_val;
        s_sum_pool[warp_id] = local.sum_exp;
    }
    __syncthreads();

    if (warp_id == 0) {
        SoftmaxPair block_res = {-1e20f, 0.0f};
        if (tid < num_warps) {
            block_res.max_val = s_max_pool[tid];
            block_res.sum_exp = s_sum_pool[tid];
        }
        warp_reduce_online(block_res);
        if (tid == 0) {
            s_final_max = block_res.max_val;
            s_final_sum = block_res.sum_exp;
        }
    }
    __syncthreads();

    const float final_max = s_final_max;
    const float final_sum = s_final_sum;
    const float inv_block_sum = __frcp_rn(final_sum);

    if (tid == 0) {
        losses[token_idx] = __logf(final_sum) + final_max - s_target_logit;
    }

    const int target_v4_idx = target_label / 4;
    const int target_v4_off = target_label % 4;

    for (int v_f4 = tid; v_f4 < vocab_size_v4; v_f4 += blockDim.x) {
        const float4 log_v4 = __ldcs(&token_logits_v4[v_f4]);
        float4 grad_v4;

        grad_v4.x = __expf(log_v4.x - final_max) * inv_block_sum;
        grad_v4.y = __expf(log_v4.y - final_max) * inv_block_sum;
        grad_v4.z = __expf(log_v4.z - final_max) * inv_block_sum;
        grad_v4.w = __expf(log_v4.w - final_max) * inv_block_sum;

        if (v_f4 == target_v4_idx) [[unlikely]] {
            if (target_v4_off == 0) grad_v4.x -= 1.0f;
            else if (target_v4_off == 1) grad_v4.y -= 1.0f;
            else if (target_v4_off == 2) grad_v4.z -= 1.0f;
            else if (target_v4_off == 3) grad_v4.w -= 1.0f;
        }

        __stcs(&out_grads_v4[v_f4], grad_v4);
    }
}

extern "C" {
void launch_cross_entropy_loss(
    const float *logits,
    float *grads,
    const int *targets,
    float *losses,
    const int total_tokens,
    const int vocab_size,
    void *stream_ptr
) {
    if (total_tokens == 0 || vocab_size == 0) return;

    int threads = 256;
    if (vocab_size > 65536) {
        threads = 1024;
    } else if (vocab_size > 32000) {
        threads = 512;
    }

    const int blocks = total_tokens;
    const int num_warps = threads / 32;
    const size_t shared_mem_size = num_warps * sizeof(float) * 2;
    const auto stream = static_cast<cudaStream_t>(stream_ptr);

    if (vocab_size % 4 == 0) [[likely]] {
        fused_cross_entropy_online_kernel<<<blocks, threads, shared_mem_size, stream>>>(
            logits, grads, targets, losses, total_tokens, vocab_size / 4, vocab_size
        );
    }
}
}
