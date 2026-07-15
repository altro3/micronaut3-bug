#include "cache_types.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

template<CacheType T>
__device__ __forceinline__ void store_cache_x4(void *base_ptr, int f4_idx, float4 vals, float scale) {
    if constexpr (T == CacheType::FP32) {
        *(static_cast<float4 *>(base_ptr) + f4_idx) = vals;
    } else if constexpr (T == CacheType::FP16) {
        int2 h2_pair;
        const auto h2_low = reinterpret_cast<half2 *>(&h2_pair.x);
        const auto h2_high = reinterpret_cast<half2 *>(&h2_pair.y);

        h2_low->x = __float2half(vals.x);
        h2_low->y = __float2half(vals.y);
        h2_high->x = __float2half(vals.z);
        h2_high->y = __float2half(vals.w);

        *(static_cast<int2 *>(base_ptr) + f4_idx) = h2_pair;
    } else if constexpr (T == CacheType::FP8) {
        uchar4 bytes;
        const float inv_scale = 1.0f / (scale + 1e-9f);

        *reinterpret_cast<__nv_fp8_e4m3 *>(&bytes.x) = __nv_fp8_e4m3(vals.x * inv_scale);
        *reinterpret_cast<__nv_fp8_e4m3 *>(&bytes.y) = __nv_fp8_e4m3(vals.y * inv_scale);
        *reinterpret_cast<__nv_fp8_e4m3 *>(&bytes.z) = __nv_fp8_e4m3(vals.z * inv_scale);
        *reinterpret_cast<__nv_fp8_e4m3 *>(&bytes.w) = __nv_fp8_e4m3(vals.w * inv_scale);

        *(static_cast<uchar4 *>(base_ptr) + f4_idx) = bytes;
    }
}

template<CacheType T>
__global__ void paged_kv_cache_write_kernel(
    void * __restrict__ k_block_table,
    void * __restrict__ v_block_table,
    const float * __restrict__ k_src,
    const float * __restrict__ v_src,
    const int32_t * __restrict__ block_mapping,
    const int32_t * __restrict__ seq_lengths,
    float * __restrict__ k_scales,
    float * __restrict__ v_scales,
    int num_seqs,
    int num_kv_heads,
    int head_dim,
    int max_blocks_per_seq,
    int block_size,
    int is_prefill
) {
    const int seq_idx = blockIdx.x;
    const int kv_head_idx = blockIdx.y;
    const int tid = threadIdx.x;

    if (seq_idx >= num_seqs || kv_head_idx >= num_kv_heads) return;

    const int head_dim_f4 = head_dim / 4;
    const int cur_seq_len = seq_lengths[seq_idx];

    const int total_toks_to_write = is_prefill ? cur_seq_len : 1;

    constexpr int bytes_per_elem = T == CacheType::FP32 ? 4 : T == CacheType::FP16 ? 2 : 1;
    const long long block_stride = static_cast<long long>(block_size) * num_kv_heads * head_dim * bytes_per_elem;
    const long long scale_block_stride = static_cast<long long>(block_size) * num_kv_heads;

    for (int t = 0; t < total_toks_to_write; ++t) {
        const int global_tok_idx = is_prefill ? t : (cur_seq_len - 1);

        const int logical_block_idx = global_tok_idx / block_size;
        const int offset_in_block = global_tok_idx % block_size;
        const int physical_block_id = block_mapping[static_cast<long long>(seq_idx) * max_blocks_per_seq + logical_block_idx];

        const long long src_row_offset = (static_cast<long long>(seq_idx) * total_toks_to_write + t) * num_kv_heads + kv_head_idx;
        const float *const k_src_ptr = k_src + src_row_offset * head_dim;
        const float *const v_src_ptr = v_src + src_row_offset * head_dim;

        const long long dst_row_offset = physical_block_id * block_stride +
                                         (static_cast<long long>(offset_in_block) * num_kv_heads + kv_head_idx) * head_dim * bytes_per_elem;
        void *const k_dst_ptr = static_cast<uint8_t *>(k_block_table) + dst_row_offset;
        void *const v_dst_ptr = static_cast<uint8_t *>(v_block_table) + dst_row_offset;

        float k_s = 1.0f;
        float v_s = 1.0f;

        if constexpr (T == CacheType::FP8) {
            float local_max_k = 0.0f;
            float local_max_v = 0.0f;

            for (int i = tid; i < head_dim; i += blockDim.x) {
                local_max_k = fmaxf(local_max_k, fabsf(k_src_ptr[i]));
                local_max_v = fmaxf(local_max_v, fabsf(v_src_ptr[i]));
            }

            for (int offset = 16; offset > 0; offset >>= 1) {
                local_max_k = fmaxf(local_max_k, __shfl_down_sync(0xFFFFFFFF, local_max_k, offset));
                local_max_v = fmaxf(local_max_v, __shfl_down_sync(0xFFFFFFFF, local_max_v, offset));
            }

            __shared__ float s_max_k;
            __shared__ float s_max_v;
            if (tid == 0) {
                s_max_k = local_max_k;
                s_max_v = local_max_v;
            }
            __syncthreads();

            k_s = s_max_k / 448.0f;
            v_s = s_max_v / 448.0f;

            if (tid == 0) {
                const long long scale_offset = physical_block_id * scale_block_stride +
                                               static_cast<long long>(offset_in_block) * num_kv_heads + kv_head_idx;
                if (k_scales) k_scales[scale_offset] = k_s;
                if (v_scales) v_scales[scale_offset] = v_s;
            }
        }

        for (int d = tid; d < head_dim_f4; d += blockDim.x) {
            const float4 k_vals = *reinterpret_cast<const float4 *>(&k_src_ptr[d * 4]);
            const float4 v_vals = *reinterpret_cast<const float4 *>(&v_src_ptr[d * 4]);

            store_cache_x4<T>(k_dst_ptr, d, k_vals, k_s);
            store_cache_x4<T>(v_dst_ptr, d, v_vals, v_s);
        }
    }
}

template __global__ void paged_kv_cache_write_kernel<CacheType::FP32>(
    void * __restrict__, void * __restrict__, const float * __restrict__, const float * __restrict__,
    const int32_t * __restrict__, const int32_t * __restrict__, float * __restrict__, float * __restrict__,
    int, int, int, int, int, int);

template __global__ void paged_kv_cache_write_kernel<CacheType::FP16>(
    void * __restrict__, void * __restrict__, const float * __restrict__, const float * __restrict__,
    const int32_t * __restrict__, const int32_t * __restrict__, float * __restrict__, float * __restrict__,
    int, int, int, int, int, int);

template __global__ void paged_kv_cache_write_kernel<CacheType::FP8>(
    void * __restrict__, void * __restrict__, const float * __restrict__, const float * __restrict__,
    const int32_t * __restrict__, const int32_t * __restrict__, float * __restrict__, float * __restrict__,
    int, int, int, int, int, int);
