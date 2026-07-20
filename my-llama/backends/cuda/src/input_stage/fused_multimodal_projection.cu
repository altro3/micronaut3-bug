#include <cuda_runtime.h>
#include <stdint.h>
#include <cassert>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include "data_types.h"

#include "cutlass/gemm/gemm.h"

template<typename T_weight>
__global__ void custom_projection_gemm_kernel(
    const __nv_bfloat16 * __restrict__ input_hidden_states,
    const void * __restrict__ projection_weights,
    __nv_bfloat16 * __restrict__ output_text_features,
    const float * __restrict__ projection_bias,
    const float * __restrict__ quantization_scales,
    const int32_t batch_num_tokens,
    const int32_t local_output_dim,
    const int32_t input_feature_dim,
    const int32_t rank_offset,
    const int32_t tile_size_m,
    const int32_t tile_size_n,
    const int32_t tile_size_k
) {
    const int32_t block_m = blockIdx.x * tile_size_m;
    const int32_t block_n = blockIdx.y * tile_size_n;

    const int32_t tid = threadIdx.x;
    const int32_t warp_id = tid / 32;
    const int32_t lane_id = tid % 32;

    extern __shared__ uint8_t dynamic_shmem[];
    const auto shmem_input = reinterpret_cast<__nv_bfloat16 *>(dynamic_shmem);
    __nv_bfloat16 *shmem_weights = shmem_input + tile_size_m * tile_size_k;

    float accum[4][4] = {0.0f};

    float scale = 1.0f;
    if (quantization_scales != nullptr) {
        scale = *quantization_scales;
    }

    for (int32_t k_offset = 0; k_offset < input_feature_dim; k_offset += tile_size_k) {
        for (int32_t i = tid; i < tile_size_m * tile_size_k; i += blockDim.x) {
            const int32_t m_local = i / tile_size_k;
            const int32_t k_local = i % tile_size_k;
            const int32_t m_global = block_m + m_local;
            const int32_t k_global = k_offset + k_local;

            if (m_global < batch_num_tokens && k_global < input_feature_dim) {
                shmem_input[m_local * tile_size_k + k_local] = __ldcs(&input_hidden_states[m_global * input_feature_dim + k_global]);
            } else {
                shmem_input[m_local * tile_size_k + k_local] = __float2bfloat16(0.0f);
            }
        }

        if constexpr (std::is_same_v<T_weight, __nv_bfloat16>) {
            const auto weights_bf16 = static_cast<const __nv_bfloat16 *>(projection_weights);
            for (int32_t i = tid; i < tile_size_n * tile_size_k; i += blockDim.x) {
                const int32_t n_local = i / tile_size_k;
                const int32_t k_local = i % tile_size_k;
                const int32_t n_global = block_n + n_local;
                const int32_t k_global = k_offset + k_local;

                if (n_global < local_output_dim && k_global < input_feature_dim) {
                    shmem_weights[n_local * tile_size_k + k_local] = __ldcs(&weights_bf16[n_global * input_feature_dim + k_global]);
                } else {
                    shmem_weights[n_local * tile_size_k + k_local] = __float2bfloat16(0.0f);
                }
            }
        } else if constexpr (std::is_same_v<T_weight, __nv_fp8_e4m3>) {
            const auto weights_fp8 = static_cast<const uint8_t *>(projection_weights);
            for (int32_t i = tid; i < tile_size_n * tile_size_k; i += blockDim.x) {
                const int32_t n_local = i / tile_size_k;
                const int32_t k_local = i % tile_size_k;
                const int32_t n_global = block_n + n_local;
                const int32_t k_global = k_offset + k_local;

                if (n_global < local_output_dim && k_global < input_feature_dim) {
                    uint8_t raw_fp8 = __ldcs(&weights_fp8[n_global * input_feature_dim + k_global]);
                    __half2_raw h2 = __nv_cvt_fp8x2_to_halfraw2(*reinterpret_cast<uint16_t *>(&raw_fp8), __NV_E4M3);
                    const float2 f2 = __half22float2(*reinterpret_cast<__half2 *>(&h2));
                    shmem_weights[n_local * tile_size_k + k_local] = __float2bfloat16(f2.x * scale);
                } else {
                    shmem_weights[n_local * tile_size_k + k_local] = __float2bfloat16(0.0f);
                }
            }
        } else if constexpr (std::is_same_v<T_weight, __nv_fp4_e2m1>) {
            const auto weights_fp4 = static_cast<const uint8_t *>(projection_weights);
            for (int32_t i = tid; i < tile_size_n * tile_size_k; i += blockDim.x) {
                const int32_t n_local = i / tile_size_k;
                const int32_t k_local = i % tile_size_k;
                const int32_t n_global = block_n + n_local;
                const int32_t k_global = k_offset + k_local;

                const int32_t global_bit_idx = (n_global * input_feature_dim + k_global) * 4;
                const int32_t global_byte_idx = global_bit_idx / 8;
                const int32_t sub_byte_offset = global_bit_idx % 8 / 4;

                if (n_global < local_output_dim && k_global < input_feature_dim) {
                    const uint8_t packed_byte = __ldcs(&weights_fp4[global_byte_idx]);
                    const uint8_t raw_fp4 = (packed_byte >> (sub_byte_offset * 4)) & 0x0F;
                    __half2_raw h2 = __nv_cvt_fp4x2_to_halfraw2(raw_fp4, __NV_E2M1);
                    const float2 f2 = __half22float2(*reinterpret_cast<__half2 *>(&h2));
                    shmem_weights[n_local * tile_size_k + k_local] = __float2bfloat16(f2.x * scale);
                } else {
                    shmem_weights[n_local * tile_size_k + k_local] = __float2bfloat16(0.0f);
                }
            }
        }

        __syncthreads();

        for (int32_t k = 0; k < tile_size_k; ++k) {
            for (int32_t m = 0; m < 4; ++m) {
                const int32_t m_local = warp_id * 16 + m;
                for (int32_t n = 0; n < 4; ++n) {
                    const int32_t n_local = lane_id + n;
                    if (m_local < tile_size_m && n_local < tile_size_n) {
                        const float input_val = __bfloat162float(shmem_input[m_local * tile_size_k + k]);
                        const float weight_val = __bfloat162float(shmem_weights[n_local * tile_size_k + k]);
                        accum[m][n] += input_val * weight_val;
                    }
                }
            }
        }

        __syncthreads();
    }

    for (int32_t m = 0; m < 4; ++m) {
        const int32_t m_global = block_m + warp_id * 16 + m;
        for (int32_t n = 0; n < 4; ++n) {
            const int32_t n_local = lane_id + n;
            const int32_t n_global = block_n + n_local;

            if (m_global < batch_num_tokens && n_global < local_output_dim) {
                const float bias_val = projection_bias != nullptr ? projection_bias[rank_offset + n_global] : 0.0f;
                float final_val = accum[m][n] + bias_val;

                if (final_val < 0.0f) {
                    final_val = final_val * 0.1702f;
                } else {
                    final_val = final_val / (1.0f + expf(-final_val * 1.702f));
                }

                __stcs(&output_text_features[m_global * local_output_dim + n_global], __float2bfloat16(final_val));
            }
        }
    }
}

extern "C" void launch_fused_multimodal_projection(
    const void ** __restrict__ host_ptr_A,
    const void ** __restrict__ host_ptr_B,
    void ** __restrict__ host_ptr_D,
    const float * __restrict__ bias,
    const float * __restrict__ weight_scales,
    const cutlass::gemm::GemmCoord * __restrict__ host_problem_shapes,
    const int32_t num_segments,
    const int32_t vision_hidden_size,
    const int32_t text_hidden_size,
    const int32_t tp_rank,
    const int32_t tp_size,
    const int32_t data_type,
    void * __restrict__ workspace_ptr,
    const cudaStream_t stream
) {
    const int32_t local_output_dim = text_hidden_size / tp_size;
    const int32_t rank_offset = tp_rank * local_output_dim;
    const auto type = static_cast<DataType>(data_type);

    constexpr int32_t TILE_M = 64;
    constexpr int32_t TILE_N = 64;
    constexpr int32_t TILE_K = 32;

    constexpr size_t shmem_size = (TILE_M * TILE_K + TILE_N * TILE_K) * sizeof(__nv_bfloat16);

    for (int32_t i = 0; i < num_segments; ++i) {
        cutlass::gemm::GemmCoord problem_size = host_problem_shapes[i];
        if (problem_size.m() <= 0) continue;

        assert(problem_size.k() == vision_hidden_size);

        const int32_t batch_num_tokens = problem_size.m();
        const int32_t input_feature_dim = problem_size.k();

        if (workspace_ptr != nullptr) {
            constexpr uintptr_t align_mask = 15;
            const auto aligned_workspace = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(workspace_ptr) + align_mask & ~align_mask);
            const auto sync_flag = static_cast<int32_t *>(aligned_workspace);
            cudaMemsetAsync(sync_flag, 0, sizeof(int32_t), stream);
        }

        dim3 block(128);
        dim3 grid((batch_num_tokens + (TILE_M - 1)) / TILE_M, (local_output_dim + (TILE_N - 1)) / TILE_N);

        const auto input_hidden_states = static_cast<const __nv_bfloat16 *>(host_ptr_A[i]);
        const void *projection_weights = host_ptr_B[i];
        const auto output_text_features = static_cast<__nv_bfloat16 *>(host_ptr_D[i]);
        const float *scale_ptr = weight_scales != nullptr ? weight_scales + i : nullptr;

        switch (type) {
            case DataType::BF16:
                custom_projection_gemm_kernel<__nv_bfloat16><<<grid, block, shmem_size, stream>>>(
                    input_hidden_states, projection_weights, output_text_features, bias, scale_ptr,
                    batch_num_tokens, local_output_dim, input_feature_dim, rank_offset, TILE_M, TILE_N, TILE_K
                );
                break;
            case DataType::FP8:
                custom_projection_gemm_kernel<__nv_fp8_e4m3><<<grid, block, shmem_size, stream>>>(
                    input_hidden_states, projection_weights, output_text_features, bias, scale_ptr,
                    batch_num_tokens, local_output_dim, input_feature_dim, rank_offset, TILE_M, TILE_N, TILE_K
                );
                break;
            case DataType::FP4:
                custom_projection_gemm_kernel<__nv_fp4_e2m1><<<grid, block, shmem_size, stream>>>(
                    input_hidden_states, projection_weights, output_text_features, bias, scale_ptr, batch_num_tokens, local_output_dim, input_feature_dim, rank_offset, TILE_M, TILE_N, TILE_K);
                break;
        }
    }
}
