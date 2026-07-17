#include "data_types.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

template<DataType T>
__device__ __forceinline__ void store_cache_x4(void *base_ptr, const int f4_idx, const float4 vals, const float scale) {
    if constexpr (T == DataType::FP32) {
        float4 *ptr = static_cast<float4 *>(base_ptr) + f4_idx;
        *ptr = vals;
    } else if constexpr (T == DataType::FP16) {
        int2 h2_pair;
        *reinterpret_cast<half2 *>(&h2_pair.x) = __floats2half2_rn(vals.x, vals.y);
        *reinterpret_cast<half2 *>(&h2_pair.y) = __floats2half2_rn(vals.z, vals.w);
        uint8_t *byte_ptr = static_cast<uint8_t *>(base_ptr) + (static_cast<size_t>(f4_idx) * sizeof(int2));
        *reinterpret_cast<int2 *>(byte_ptr) = h2_pair;
    } else if constexpr (T == DataType::FP8) {
        const float inv_scale = 1.0f / (scale + 1e-9f);

        const half2 h2_low = __floats2half2_rn(vals.x * inv_scale, vals.y * inv_scale);
        const half2 h2_high = __floats2half2_rn(vals.z * inv_scale, vals.w * inv_scale);

        uint32_t res;
        *reinterpret_cast<__nv_fp8x2_e4m3 *>(&res) = __nv_fp8x2_e4m3(h2_low);
        *(reinterpret_cast<__nv_fp8x2_e4m3 *>(&res) + 1) = __nv_fp8x2_e4m3(h2_high);

        uint8_t *byte_ptr = static_cast<uint8_t *>(base_ptr) + (static_cast<size_t>(f4_idx) * sizeof(uint32_t));
        *reinterpret_cast<uint32_t *>(byte_ptr) = res;
    }
}

template<DataType T>
__global__ void write_kv_to_paged_cache_kernel_optimized(
    const float * __restrict__ src_key_states,
    const float * __restrict__ src_value_states,
    const int32_t * __restrict__ global_slot_mapping,
    void * __restrict__ dst_paged_key_cache,
    void * __restrict__ dst_paged_value_cache,
    float * __restrict__ k_scales,
    float * __restrict__ v_scales,
    const int32_t total_batch_tokens,
    const int32_t num_kv_heads,
    const int32_t head_dimension,
    const int32_t block_size
) {
    const int token_index = blockIdx.x;
    const int kv_head_index = blockIdx.y;

    if (token_index >= total_batch_tokens || kv_head_index >= num_kv_heads) return;

    const int tid = threadIdx.x;
    const int total_threads = blockDim.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int head_dimension_f4 = head_dimension / 4;
    const int num_warps = (total_threads + 31) / 32;

    const int32_t target_physical_slot = global_slot_mapping[token_index];
    const int32_t physical_block_id = target_physical_slot / block_size;
    const int32_t offset_in_block = target_physical_slot % block_size;

    constexpr int bytes_per_element = T == DataType::FP32 ? 4 : T == DataType::FP16 ? 2 : 1;
    const long long block_stride = static_cast<long long>(block_size) * num_kv_heads * head_dimension * bytes_per_element;
    const long long scale_block_stride = static_cast<long long>(block_size) * num_kv_heads;

    extern __shared__ float s_shared_dynamic_buffer[];
    float *s_warp_max_k = s_shared_dynamic_buffer;
    float *s_warp_max_v = s_shared_dynamic_buffer + num_warps;

    const long long src_token_offset = (static_cast<long long>(token_index) * num_kv_heads + kv_head_index) * head_dimension;
    const float *const src_k_ptr = src_key_states + src_token_offset;
    const float *const src_v_ptr = src_value_states + src_token_offset;

    const long long dst_memory_offset = physical_block_id * block_stride
                                        + (static_cast<long long>(offset_in_block) * num_kv_heads + kv_head_index) * head_dimension * bytes_per_element;
    void *const dst_k_ptr = static_cast<uint8_t *>(dst_paged_key_cache) + dst_memory_offset;
    void *const dst_v_ptr = static_cast<uint8_t *>(dst_paged_value_cache) + dst_memory_offset;

    float k_s = 1.0f;
    float v_s = 1.0f;

    if constexpr (T == DataType::FP8) {
        float local_max_k = 0.0f;
        float local_max_v = 0.0f;

        for (int i = tid; i < head_dimension; i += total_threads) {
            local_max_k = fmaxf(local_max_k, fabsf(src_k_ptr[i]));
            local_max_v = fmaxf(local_max_v, fabsf(src_v_ptr[i]));
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            local_max_k = fmaxf(local_max_k, __shfl_down_sync(0xFFFFFFFF, local_max_k, offset));
            local_max_v = fmaxf(local_max_v, __shfl_down_sync(0xFFFFFFFF, local_max_v, offset));
        }

        if (lane_id == 0) {
            s_warp_max_k[warp_id] = local_max_k;
            s_warp_max_v[warp_id] = local_max_v;
        }
        __syncthreads();

        if (warp_id == 0) {
            float block_max_k = (lane_id < num_warps) ? s_warp_max_k[lane_id] : 0.0f;
            float block_max_v = (lane_id < num_warps) ? s_warp_max_v[lane_id] : 0.0f;

            for (int offset = 16; offset > 0; offset >>= 1) {
                block_max_k = fmaxf(block_max_k, __shfl_down_sync(0xFFFFFFFF, block_max_k, offset));
                block_max_v = fmaxf(block_max_v, __shfl_down_sync(0xFFFFFFFF, block_max_v, offset));
            }

            if (lane_id == 0) {
                s_warp_max_k[0] = block_max_k / 448.0f;
                s_warp_max_v[0] = block_max_v / 448.0f;

                const long long scale_offset = physical_block_id * scale_block_stride +
                                               static_cast<long long>(offset_in_block) * num_kv_heads + kv_head_index;
                if (k_scales) k_scales[scale_offset] = s_warp_max_k[0];
                if (v_scales) v_scales[scale_offset] = s_warp_max_v[0];
            }
        }
        __syncthreads();

        k_s = s_warp_max_k[0];
        v_s = s_warp_max_v[0];
    }

    for (int d = tid; d < head_dimension_f4; d += total_threads) {
        const float4 k_vals = *reinterpret_cast<const float4 *>(&src_k_ptr[d * 4]);
        const float4 v_vals = *reinterpret_cast<const float4 *>(&src_v_ptr[d * 4]);

        store_cache_x4<T>(dst_k_ptr, d, k_vals, k_s);
        store_cache_x4<T>(dst_v_ptr, d, v_vals, v_s);
    }
}

extern "C" {
void launch_prefill_kv_write(
    void *dst_paged_key_cache,
    void *dst_paged_value_cache,
    const float *src_key_states,
    const float *src_value_states,
    const int32_t *global_slot_mapping,
    float *k_scales,
    float *v_scales,
    const int data_type_id,
    const int total_batch_tokens,
    const int num_kv_heads,
    const int head_dimension,
    const int block_size,
    int threads_per_block,
    void *stream_ptr
) {
    if (total_batch_tokens == 0) return;

    dim3 grid(total_batch_tokens, num_kv_heads);
    const auto stream = static_cast<cudaStream_t>(stream_ptr);
    const int num_warps = (threads_per_block + 31) / 32;
    const int shared_mem_size = num_warps * 2 * sizeof(float);

    if (data_type_id == static_cast<int>(DataType::FP32)) {
        write_kv_to_paged_cache_kernel_optimized<DataType::FP32> <<<grid, threads_per_block, shared_mem_size, stream>>>(
            src_key_states, src_value_states, global_slot_mapping, dst_paged_key_cache, dst_paged_value_cache,
            k_scales, v_scales, total_batch_tokens, num_kv_heads, head_dimension, block_size
        );
    } else if (data_type_id == static_cast<int>(DataType::FP16)) {
        write_kv_to_paged_cache_kernel_optimized<DataType::FP16> <<<grid, threads_per_block, shared_mem_size, stream>>>(
            src_key_states, src_value_states, global_slot_mapping, dst_paged_key_cache, dst_paged_value_cache,
            k_scales, v_scales, total_batch_tokens, num_kv_heads, head_dimension, block_size
        );
    } else if (data_type_id == static_cast<int>(DataType::FP8)) {
        write_kv_to_paged_cache_kernel_optimized<DataType::FP8> <<<grid, threads_per_block, shared_mem_size, stream>>>(
            src_key_states, src_value_states, global_slot_mapping, dst_paged_key_cache, dst_paged_value_cache,
            k_scales, v_scales, total_batch_tokens, num_kv_heads, head_dimension, block_size
        );
    }
}
}
