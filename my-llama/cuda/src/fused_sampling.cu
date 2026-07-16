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
        const __half2 *h2_ptr = reinterpret_cast<const __half2 *>(&vec);
        const float2 low = __half22float2(h2_ptr[0]);
        const float2 high = __half22float2(h2_ptr[1]);
        out_vals[0] = low.x;
        out_vals[1] = low.y;
        out_vals[2] = high.x;
        out_vals[3] = high.y;
    } else if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
        const uint32_t vec = *(static_cast<const uint32_t *>(logits) + base_index);
        const uint8_t *bytes = reinterpret_cast<const uint8_t *>(&vec);

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
__global__ void fused_sampling_kernel(
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

    float local_max = -INFINITY;
    const int vocab_size_v4 = vocab_size / 4;

    for (int i = tid; i < vocab_size_v4; i += blockDim.x) {
        float vals[4];
        fetch_logits_x4<T>(logits, i, vals);
        local_max = fmaxf(local_max, fmaxf(fmaxf(vals[0], vals[1]), fmaxf(vals[2], vals[3])));
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        const float shuffled = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
        local_max = fmaxf(local_max, shuffled);
    }

    __shared__ float s_max_logit;
    if (tid == 0) {
        s_max_logit = local_max;
    }
    __syncthreads();

    const float max_logit = s_max_logit;
    const float inv_temp = 1.0f / (temperature + 1e-9f);

    float warp_topk_vals[MAX_K];
    int warp_topk_ids[MAX_K];
    for (int k = 0; k < MAX_K; ++k) {
        warp_topk_vals[k] = -INFINITY;
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
                        const float temp_v = warp_topk_vals[k];
                        warp_topk_vals[k] = warp_topk_vals[k - 1];
                        warp_topk_vals[k - 1] = temp_v;

                        const int temp_id = warp_topk_ids[k];
                        warp_topk_ids[k] = warp_topk_ids[k - 1];
                        warp_topk_ids[k - 1] = temp_id;
                    } else {
                        break;
                    }
                }
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        for (int k = 0; k < MAX_K; ++k) {
            const float remote_val = __shfl_down_sync(0xFFFFFFFF, warp_topk_vals[k], offset);
            const int remote_id = __shfl_down_sync(0xFFFFFFFF, warp_topk_ids[k], offset);

            if (lane_id < offset) {
                if (remote_val > warp_topk_vals[MAX_K - 1]) {
                    warp_topk_vals[MAX_K - 1] = remote_val;
                    warp_topk_ids[MAX_K - 1] = remote_id;

                    for (int m = MAX_K - 1; m > 0; --m) {
                        if (warp_topk_vals[m] > warp_topk_vals[m - 1]) {
                            const float temp_v = warp_topk_vals[m];
                            warp_topk_vals[m] = warp_topk_vals[m - 1];
                            warp_topk_vals[m - 1] = temp_v;

                            const int temp_id = warp_topk_ids[m];
                            warp_topk_ids[m] = warp_topk_ids[m - 1];
                            warp_topk_ids[m - 1] = temp_id;
                        } else {
                            break;
                        }
                    }
                }
            }
        }
    }

    __shared__ float s_block_topk_vals[8][MAX_K];
    __shared__ int s_block_topk_ids[8][MAX_K];

    if (lane_id == 0) {
        for (int k = 0; k < MAX_K; ++k) {
            s_block_topk_vals[warp_id][k] = warp_topk_vals[k];
            s_block_topk_ids[warp_id][k] = warp_topk_ids[k];
        }
    }
    __syncthreads();

    if (tid == 0) {
        float final_topk_vals[MAX_K];
        int final_topk_ids[MAX_K];
        for (int k = 0; k < MAX_K; ++k) {
            final_topk_vals[k] = -INFINITY;
            final_topk_ids[k] = -1;
        }

        const int active_warps = blockDim.x / 32;
        for (int w = 0; w < active_warps; ++w) {
            for (int i = 0; i < MAX_K; ++i) {
                const float prob = s_block_topk_vals[w][i];
                const int id = s_block_topk_ids[w][i];

                if (prob > final_topk_vals[MAX_K - 1]) {
                    final_topk_vals[MAX_K - 1] = prob;
                    final_topk_ids[MAX_K - 1] = id;

                    for (int k = MAX_K - 1; k > 0; --k) {
                        if (final_topk_vals[k] > final_topk_vals[k - 1]) {
                            const float temp_v = final_topk_vals[k];
                            final_topk_vals[k] = final_topk_vals[k - 1];
                            final_topk_vals[k - 1] = temp_v;

                            const int temp_id = final_topk_ids[k];
                            final_topk_ids[k] = final_topk_ids[k - 1];
                            final_topk_ids[k - 1] = temp_id;
                        } else {
                            break;
                        }
                    }
                }
            }
        }

        float sum_probs = 0.0f;
        for (int i = 0; i < MAX_K; ++i) {
            if (final_topk_ids[i] != -1) {
                sum_probs += final_topk_vals[i];
            }
        }

        const float inv_sum = 1.0f / (sum_probs + 1e-9f);
        for (int i = 0; i < MAX_K; ++i) {
            if (final_topk_ids[i] != -1) {
                final_topk_vals[i] *= inv_sum;
            }
        }

        float cumulative_prob = 0.0f;
        int last_valid_idx = 0;
        for (int i = 0; i < MAX_K; ++i) {
            if (final_topk_ids[i] == -1) break;
            cumulative_prob += final_topk_vals[i];
            last_valid_idx = i;
            if (cumulative_prob >= top_p) {
                break;
            }
        }

        float truncated_sum = 0.0f;
        for (int i = 0; i <= last_valid_idx; ++i) {
            truncated_sum += final_topk_vals[i];
        }

        const float inv_truncated_sum = 1.0f / (truncated_sum + 1e-9f);
        float current_target = rand_val;
        int selected_token = final_topk_ids[0];

        for (int i = 0; i <= last_valid_idx; ++i) {
            const float norm_prob = final_topk_vals[i] * inv_truncated_sum;
            if (current_target <= norm_prob) {
                selected_token = final_topk_ids[i];
                break;
            }
            current_target -= norm_prob;
        }

        if (selected_token == -1) {
            selected_token = final_topk_ids[0];
        }

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

    if (data_type_id == static_cast<int>(DataType::FP32)) {
        fused_sampling_kernel<float> <<<1, threads_per_block, 0, stream>>>(
            token_id, logits, rand_val, temperature, top_p, vocab_size
        );
    } else if (data_type_id == static_cast<int>(DataType::FP16)) {
        fused_sampling_kernel<__half> <<<1, threads_per_block, 0, stream>>>(
            token_id, logits, rand_val, temperature, top_p, vocab_size
        );
    } else if (data_type_id == static_cast<int>(DataType::FP8)) {
        fused_sampling_kernel<__nv_fp8_e4m3> <<<1, threads_per_block, 0, stream>>>(
            token_id, logits, rand_val, temperature, top_p, vocab_size
        );
    }
}
}
