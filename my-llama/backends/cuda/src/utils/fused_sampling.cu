#include "data_types.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <math.h>

#define MAX_K 64

template<typename T>
__device__ __forceinline__ void fetch_logits_x4(const void * __restrict__ logits, const int base_index, float *out_vals) {
    if constexpr (std::is_same_v<T, float>) {
        const float4 vec = *(static_cast<const float4 *>(logits) + base_index);
        out_vals[0] = vec.x;
        out_vals[1] = vec.y;
        out_vals[2] = vec.z;
        out_vals[3] = vec.w;
    } else if constexpr (std::is_same_v<T, __half>) {
        const int2 vec = *(static_cast<const int2 *>(logits) + base_index);
        const auto h2_ptr = reinterpret_cast<const __half2 *>(&vec);
        const float2 low = __half22float2(h2_ptr[0]);
        const float2 high = __half22float2(h2_ptr[1]);
        out_vals[0] = low.x;
        out_vals[1] = low.y;
        out_vals[2] = high.x;
        out_vals[3] = high.y;
    } else if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
        const uint32_t vec = *(static_cast<const uint32_t *>(logits) + base_index);
        const auto bytes = reinterpret_cast<const uint8_t *>(&vec);

        __nv_fp8_e4m3 r0, r1, r2, r3;
        r0.__x = bytes[0];
        r1.__x = bytes[1];
        r2.__x = bytes[2];
        r3.__x = bytes[3];

        out_vals[0] = __half2float(__half(r0));
        out_vals[1] = __half2float(__half(r1));
        out_vals[2] = __half2float(__half(r2));
        out_vals[3] = __half2float(__half(r3));
    }
}

template<typename T>
__global__ void fused_sampling_kernel_optimized(
    int * __restrict__ token_id,
    const void * __restrict__ logits,
    const float rand_val,
    const float temperature,
    const float top_p,
    const int vocab_size
) {
    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int num_warps = (blockDim.x + 31) / 32;

    float local_max = -INFINITY;
    const int vocab_size_v4 = vocab_size / 4;

    for (int i = tid; i < vocab_size_v4; i += blockDim.x) {
        float vals[4];
        fetch_logits_x4<T>(logits, i, vals);
        local_max = fmaxf(local_max, fmaxf(fmaxf(vals[0], vals[1]), fmaxf(vals[2], vals[3])));
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
    }

    extern __shared__ uint8_t s_dynamic_mem[];

    const auto s_shared_max = reinterpret_cast<float *>(s_dynamic_mem);

    const int num_warps_aligned = num_warps + 3 & ~3;
    float *s_block_vals = s_shared_max + num_warps_aligned;
    const auto s_block_ids = reinterpret_cast<int *>(s_block_vals + num_warps_aligned * MAX_K);

    const auto s_final_vals = reinterpret_cast<float *>(s_block_ids + num_warps_aligned * MAX_K);
    const auto s_final_ids = reinterpret_cast<int *>(s_final_vals + MAX_K * 2);

    if (lane_id == 0) {
        s_shared_max[warp_id] = local_max;
    }
    __syncthreads();

    if (tid == 0) {
        float block_max = s_shared_max[0];
        for (int w = 1; w < num_warps; ++w) {
            block_max = fmaxf(block_max, s_shared_max[w]);
        }
        s_shared_max[0] = block_max;
    }
    __syncthreads();

    const float max_logit = s_shared_max[0];
    const float inv_temp = 1.0f / (temperature + 1e-9f);

    float warp_topk_vals[MAX_K];
    int warp_topk_ids[MAX_K];
    for (int k = 0; k < MAX_K; ++k) {
        warp_topk_vals[k] = 0.0f;
        warp_topk_ids[k] = -1;
    }

    for (int i = tid; i < vocab_size_v4; i += blockDim.x) {
        float vals[4];
        fetch_logits_x4<T>(logits, i, vals);

        for (int v = 0; v < 4; ++v) {
            const float val = (vals[v] - max_logit) * inv_temp;
            const float prob = expf(val);
            const int global_idx = i * 4 + v;

            if (prob > warp_topk_vals[MAX_K - 1]) {
                warp_topk_vals[MAX_K - 1] = prob;
                warp_topk_ids[MAX_K - 1] = global_idx;

                for (int k = MAX_K - 1; k > 0; --k) {
                    if (warp_topk_vals[k] > warp_topk_vals[k - 1]) {
                        const float tv = warp_topk_vals[k];
                        warp_topk_vals[k] = warp_topk_vals[k - 1];
                        warp_topk_vals[k - 1] = tv;

                        const int tidx = warp_topk_ids[k];
                        warp_topk_ids[k] = warp_topk_ids[k - 1];
                        warp_topk_ids[k - 1] = tidx;
                    } else {
                        break;
                    }
                }
            }
        }
    }

    if (lane_id == 0) {
        for (int k = 0; k < MAX_K; ++k) {
            s_block_vals[warp_id * MAX_K + k] = warp_topk_vals[k];
            s_block_ids[warp_id * MAX_K + k] = warp_topk_ids[k];
        }
    }
    __syncthreads();

    if (tid == 0) {
        for (int k = 0; k < MAX_K; ++k) {
            s_final_vals[k] = 0.0f;
            s_final_ids[k] = -1;
        }

        for (int w = 0; w < num_warps; ++w) {
            for (int i = 0; i < MAX_K; ++i) {
                const float prob = s_block_vals[w * MAX_K + i];
                const int id = s_block_ids[w * MAX_K + i];

                if (id == -1) break;

                if (prob > s_final_vals[MAX_K - 1]) {
                    s_final_vals[MAX_K - 1] = prob;
                    s_final_ids[MAX_K - 1] = id;

                    for (int k = MAX_K - 1; k > 0; --k) {
                        if (s_final_vals[k] > s_final_vals[k - 1]) {
                            const float tv = s_final_vals[k];
                            s_final_vals[k] = s_final_vals[k - 1];
                            s_final_vals[k - 1] = tv;

                            const int ti = s_final_ids[k];
                            s_final_ids[k] = s_final_ids[k - 1];
                            s_final_ids[k - 1] = ti;
                        } else {
                            break;
                        }
                    }
                }
            }
        }
    }
    __syncthreads();

    if (tid < MAX_K) {
        float total_sum = 0.0f;
        for (int i = 0; i < MAX_K; ++i) {
            if (s_final_ids[i] != -1) total_sum += s_final_vals[i];
        }

        const float inv_sum = 1.0f / (total_sum + 1e-9f);
        if (s_final_ids[tid] != -1) {
            s_final_vals[tid] *= inv_sum;
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        float scan_sum = s_final_vals[lane_id];
        for (int offset = 1; offset < 32; offset <<= 1) {
            const float remote = __shfl_up_sync(0xFFFFFFFF, scan_sum, offset);
            if (lane_id >= offset) scan_sum += remote;
        }

        const float v32 = __shfl_sync(0xFFFFFFFF, scan_sum, 31);

        float scan_sum_high = s_final_vals[32 + lane_id];
        for (int offset = 1; offset < 32; offset <<= 1) {
            const float remote = __shfl_up_sync(0xFFFFFFFF, scan_sum_high, offset);
            if (lane_id >= offset) scan_sum_high += remote;
        }

        s_final_vals[MAX_K + lane_id] = scan_sum;
        s_final_vals[MAX_K + 32 + lane_id] = v32 + scan_sum_high;
    }
    __syncthreads();

    if (tid == 0) {
        const float *prefix_sums = s_final_vals + MAX_K;
        int last_valid_idx = 0;
        for (int i = 0; i < MAX_K; ++i) {
            if (s_final_ids[i] == -1) break;
            last_valid_idx = i;
            if (prefix_sums[i] >= top_p) break;
        }

        const float truncated_sum = prefix_sums[last_valid_idx];
        const float inv_truncated_sum = 1.0f / (truncated_sum + 1e-9f);
        float current_target = rand_val;
        int selected_token = s_final_ids[0];

        for (int i = 0; i <= last_valid_idx; ++i) {
            const float norm_prob = (i == 0 ? prefix_sums[0] : prefix_sums[i] - prefix_sums[i - 1]) * inv_truncated_sum;
            if (current_target <= norm_prob) {
                selected_token = s_final_ids[i];
                break;
            }
            current_target -= norm_prob;
        }

        if (selected_token == -1) selected_token = s_final_ids[0];
        *token_id = selected_token;
    }
}

extern "C" {
void launch_fused_sampling(
    int *token_id,
    const void *logits,
    const float rand_val,
    const float temperature,
    const float top_p,
    const int vocab_size,
    const int data_type_id,
    int threads_per_block,
    void *stream_ptr
) {
    if (threads_per_block <= 0) return;

    const auto stream = static_cast<cudaStream_t>(stream_ptr);
    const int num_warps = (threads_per_block + 31) / 32;
    const int num_warps_aligned = num_warps + 3 & ~3;

    const int max_size = num_warps_aligned * sizeof(float);
    const int vals_size = num_warps_aligned * MAX_K * sizeof(float);
    const int ids_size = num_warps_aligned * MAX_K * sizeof(int);
    constexpr int final_sort_vals_size = MAX_K * sizeof(float) * 2;
    constexpr int final_sort_ids_size = MAX_K * sizeof(int);

    const int shared_mem_size = max_size + vals_size + ids_size + final_sort_vals_size + final_sort_ids_size;

    if (data_type_id == static_cast<int>(DataTypeOld::FP32)) {
        fused_sampling_kernel_optimized<float> <<<1, threads_per_block, shared_mem_size, stream>>>(
            token_id, logits, rand_val, temperature, top_p, vocab_size
        );
    } else if (data_type_id == static_cast<int>(DataTypeOld::FP16)) {
        fused_sampling_kernel_optimized<__half> <<<1, threads_per_block, shared_mem_size, stream>>>(
            token_id, logits, rand_val, temperature, top_p, vocab_size
        );
    } else if (data_type_id == static_cast<int>(DataTypeOld::FP8)) {
        fused_sampling_kernel_optimized<__nv_fp8_e4m3> <<<1, threads_per_block, shared_mem_size, stream>>>(
            token_id, logits, rand_val, temperature, top_p, vocab_size
        );
    }
}
}
