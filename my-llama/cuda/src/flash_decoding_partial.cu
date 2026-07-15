#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

enum class CacheType {
    FP32 = 0,
    FP16 = 1,
    FP8 = 2
};

__device__ __forceinline__ float warp_reduce_max_fd(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum_fd(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

template<CacheType T>
__device__ __forceinline__ float4 load_cache_x4(const void *base_ptr, int f4_idx, float scale) {
    float4 f4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    if constexpr (T == CacheType::FP32) {
        f4 = __ldcs(static_cast<const float4 *>(base_ptr) + f4_idx);
    } else if constexpr (T == CacheType::FP16) {
        const int2 h2_pair = __ldcs(static_cast<const int2 *>(base_ptr) + f4_idx);
        const half2 h2_low = *reinterpret_cast<const half2 *>(&h2_pair.x);
        const half2 h2_high = *reinterpret_cast<const half2 *>(&h2_pair.y);
        f4.x = __half2float(h2_low.x);
        f4.y = __half2float(h2_low.y);
        f4.z = __half2float(h2_high.x);
        f4.w = __half2float(h2_high.y);
    } else if constexpr (T == CacheType::FP8) {
        const uchar4 bytes = __ldcs(static_cast<const uchar4 *>(base_ptr) + f4_idx);
        f4.x = static_cast<float>(*reinterpret_cast<const __nv_fp8_e4m3 *>(&bytes.x)) * scale;
        f4.y = static_cast<float>(*reinterpret_cast<const __nv_fp8_e4m3 *>(&bytes.y)) * scale;
        f4.z = static_cast<float>(*reinterpret_cast<const __nv_fp8_e4m3 *>(&bytes.z)) * scale;
        f4.w = static_cast<float>(*reinterpret_cast<const __nv_fp8_e4m3 *>(&bytes.w)) * scale;
    }
    return f4;
}

template<CacheType T>
__global__ void flash_decoding_partial_kernel(
    float * __restrict__ partial_out,
    float * __restrict__ partial_max,
    float * __restrict__ partial_sum,
    const float * __restrict__ query,
    const void * __restrict__ k_cache,
    const void * __restrict__ v_cache,
    const float * __restrict__ k_scales,
    const float * __restrict__ v_scales,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int current_seq_len,
    int chunk_size,
    int num_chunks
) {
    const int head_idx = blockIdx.x;
    const int chunk_idx = blockIdx.y;
    const int tid = threadIdx.x;

    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int num_warps = blockDim.x / 32;

    const int kv_head_ratio = num_heads / num_kv_heads;
    const int kv_head_idx = head_idx / kv_head_ratio;

    const int head_dim_f4 = head_dim / 4;
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    const int start_tok = chunk_idx * chunk_size;
    const int end_tok = min(start_tok + chunk_size, current_seq_len);

    if (start_tok >= end_tok) return;

    __shared__ float s_warp_max[32];
    __shared__ float s_warp_sum[32];
    extern __shared__ float s_shared_mem[];

    float *s_q_vec = s_shared_mem;
    float *s_scores = s_shared_mem + head_dim;

    const float *const q_vec = query + head_idx * head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        s_q_vec[i] = q_vec[i];
    }
    __syncthreads();

    float local_max = -1e20f;
    const int total_toks_to_process = end_tok - start_tok;

    constexpr int bytes_per_elem = T == CacheType::FP32 ? 4 : T == CacheType::FP16 ? 2 : 1;

    for (int local_tok = tid; local_tok < total_toks_to_process; local_tok += blockDim.x) {
        const int global_tok = start_tok + local_tok;

        const long long cache_row_offset = (static_cast<long long>(global_tok) * num_kv_heads + kv_head_idx) * head_dim * bytes_per_elem;
        const void *const k_vec_ptr = static_cast<const uint8_t *>(k_cache) + cache_row_offset;

        const long long scale_offset = static_cast<long long>(global_tok) * num_kv_heads + kv_head_idx;
        const float k_s = T == CacheType::FP8 && k_scales ? k_scales[scale_offset] : 1.0f;

        float score_acc = 0.0f;

        for (int i = 0; i < head_dim_f4; ++i) {
            const float4 q_val = *reinterpret_cast<const float4 *>(&s_q_vec[i * 4]);
            const float4 k_val = load_cache_x4<T>(k_vec_ptr, i, k_s);

            score_acc += q_val.x * k_val.x + q_val.y * k_val.y + q_val.z * k_val.z + q_val.w * k_val.w;
        }

        const float raw_score = score_acc * scale;
        s_scores[local_tok] = raw_score;
        local_max = fmaxf(local_max, raw_score);
    }

    float block_max = warp_reduce_max_fd(local_max);
    if (lane_id == 0) s_warp_max[warp_id] = block_max;
    __syncthreads();

    if (warp_id == 0) {
        const float val = tid < num_warps ? s_warp_max[tid] : -1e20f;
        block_max = warp_reduce_max_fd(val);
        if (tid == 0) s_warp_max[0] = block_max;
    }
    __syncthreads();
    block_max = s_warp_max[0];

    float local_sum = 0.0f;
    for (int local_tok = tid; local_tok < total_toks_to_process; local_tok += blockDim.x) {
        const float exp_score = expf(s_scores[local_tok] - block_max);
        s_scores[local_tok] = exp_score;
        local_sum += exp_score;
    }

    float block_sum = warp_reduce_sum_fd(local_sum);
    if (lane_id == 0) s_warp_sum[warp_id] = block_sum;
    __syncthreads();

    if (warp_id == 0) {
        const float val = tid < num_warps ? s_warp_sum[tid] : 0.0f;
        block_sum = warp_reduce_sum_fd(val);
        if (tid == 0) s_warp_sum[0] = block_sum;
    }
    __syncthreads();
    block_sum = s_warp_sum[0];

    for (int local_tok = tid; local_tok < total_toks_to_process; local_tok += blockDim.x) {
        s_scores[local_tok] /= block_sum + 1e-9f;
    }
    __syncthreads();

    const long long partial_offset = static_cast<long long>(head_idx) * num_chunks + chunk_idx;
    float *const p_out_ptr = partial_out + partial_offset * head_dim;

    for (int d = tid; d < head_dim_f4; d += blockDim.x) {
        float4 v_acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        for (int local_tok = 0; local_tok < total_toks_to_process; ++local_tok) {
            const float prob = s_scores[local_tok];
            const int global_tok = start_tok + local_tok;

            const long long v_cache_row_offset = (static_cast<long long>(global_tok) * num_kv_heads + kv_head_idx) * head_dim * bytes_per_elem;
            const void *const v_vec_ptr = static_cast<const uint8_t *>(v_cache) + v_cache_row_offset;

            const long long scale_offset = static_cast<long long>(global_tok) * num_kv_heads + kv_head_idx;
            const float v_s = T == CacheType::FP8 && v_scales ? v_scales[scale_offset] : 1.0f;

            const float4 v_val = load_cache_x4<T>(v_vec_ptr, d, v_s);

            v_acc.x += prob * v_val.x;
            v_acc.y += prob * v_val.y;
            v_acc.z += prob * v_val.z;
            v_acc.w += prob * v_val.w;
        }

        *reinterpret_cast<float4 *>(&p_out_ptr[d * 4]) = v_acc;
    }

    if (tid == 0) {
        partial_max[partial_offset] = block_max;
        partial_sum[partial_offset] = block_sum;
    }
}

template __global__ void flash_decoding_partial_kernel<CacheType::FP32>(
    float * __restrict__ partial_out,
    float * __restrict__ partial_max,
    float * __restrict__ partial_sum,
    const float * __restrict__ query,
    const void * __restrict__ k_cache,
    const void * __restrict__ v_cache,
    const float * __restrict__ k_scales,
    const float * __restrict__ v_scales,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int current_seq_len,
    int chunk_size,
    int num_chunks
);

template __global__ void flash_decoding_partial_kernel<CacheType::FP16>(
    float * __restrict__ partial_out,
    float * __restrict__ partial_max,
    float * __restrict__ partial_sum,
    const float * __restrict__ query,
    const void * __restrict__ k_cache,
    const void * __restrict__ v_cache,
    const float * __restrict__ k_scales,
    const float * __restrict__ v_scales,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int current_seq_len,
    int chunk_size,
    int num_chunks
);

template __global__ void flash_decoding_partial_kernel<CacheType::FP8>(
    float * __restrict__ partial_out,
    float * __restrict__ partial_max,
    float * __restrict__ partial_sum,
    const float * __restrict__ query,
    const void * __restrict__ k_cache,
    const void * __restrict__ v_cache,
    const float * __restrict__ k_scales,
    const float * __restrict__ v_scales,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int current_seq_len,
    int chunk_size,
    int num_chunks
);
