#include "fused_multimodal_projection.cuh"
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <type_traits>
#include <mma.h>

using namespace nvcuda::wmma;

template<typename T_weight>
__global__ void batched_projection_gemm_kernel(
    const void ** __restrict__ device_table_A,
    const void ** __restrict__ device_table_B,
    void ** __restrict__ device_table_D,
    const cutlass::gemm::GemmCoord * __restrict__ device_shapes,
    const float * __restrict__ projection_bias,
    const float * __restrict__ quantization_scales,
    int32_t local_output_dim,
    int32_t input_feature_dim,
    int32_t rank_offset,
    int32_t tile_size_m,
    int32_t tile_size_n,
    int32_t tile_size_k
) {
    const int32_t segment_id = blockIdx.z;

    cutlass::gemm::GemmCoord problem_size = device_shapes[segment_id];
    const int32_t batch_num_tokens = problem_size.m();

    if (batch_num_tokens <= 0) return;

    const int32_t block_m = blockIdx.x * tile_size_m;
    const int32_t block_n = blockIdx.y * tile_size_n;

    if (block_m >= batch_num_tokens) return;

    auto input_hidden_states = static_cast<const __nv_bfloat16 *>(device_table_A[segment_id]);
    const void *projection_weights = device_table_B[segment_id];
    auto output_text_features = static_cast<__nv_bfloat16 *>(device_table_D[segment_id]);

    const int32_t tid = threadIdx.x;
    const int32_t warp_id = tid / 32;

    extern __shared__ uint8_t dynamic_shmem[];
    auto shmem_input = reinterpret_cast<__nv_bfloat16 *>(dynamic_shmem);
    __nv_bfloat16 *shmem_weights = shmem_input + tile_size_m * tile_size_k;

    __shared__ float shared_scale;
    if (tid == 0) {
        shared_scale = (quantization_scales != nullptr) ? quantization_scales[segment_id] : 1.0f;
    }
    __syncthreads();
    float scale = shared_scale;

    fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag[2][2];

#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    for (int32_t k_offset = 0; k_offset < input_feature_dim; k_offset += tile_size_k) {
        for (int32_t i = tid; i < (tile_size_m * tile_size_k) / 8; i += blockDim.x) {
            uint4 *dest = reinterpret_cast<uint4 *>(shmem_input) + i;
            const int32_t total_elements = i * 8;
            const int32_t m_local = total_elements / tile_size_k;
            const int32_t k_local = total_elements % tile_size_k;
            const int32_t m_global = block_m + m_local;
            const int32_t k_global = k_offset + k_local;

            if (m_global < batch_num_tokens && k_global < input_feature_dim) {
                *dest = *reinterpret_cast<const uint4 *>(&input_hidden_states[m_global * input_feature_dim + k_global]);
            } else {
                *dest = make_uint4(0, 0, 0, 0);
            }
        }

        if constexpr (std::is_same_v<T_weight, __nv_bfloat16>) {
            const auto weights_bf16 = static_cast<const __nv_bfloat16 *>(projection_weights);
            for (int32_t i = tid; i < (tile_size_n * tile_size_k) / 8; i += blockDim.x) {
                uint4 *dest = reinterpret_cast<uint4 *>(shmem_weights) + i;
                const int32_t total_elements = i * 8;
                const int32_t n_local = total_elements / tile_size_k;
                const int32_t k_local = total_elements % tile_size_k;
                const int32_t n_global = block_n + n_local;
                const int32_t k_global = k_offset + k_local;

                if (n_global < local_output_dim && k_global < input_feature_dim) {
                    *dest = *reinterpret_cast<const uint4 *>(&weights_bf16[n_global * input_feature_dim + k_global]);
                } else {
                    *dest = make_uint4(0, 0, 0, 0);
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
                    uint8_t raw_fp8 = weights_fp8[n_global * input_feature_dim + k_global];
                    uint16_t packed_fp8x2 = (static_cast<uint16_t>(raw_fp8) << 8) | raw_fp8;
                    __half2_raw h2 = __nv_cvt_fp8x2_to_halfraw2(packed_fp8x2, __NV_E4M3);
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

                if (n_global < local_output_dim && k_global < input_feature_dim) {
                    const int32_t global_bit_idx = (n_global * input_feature_dim + k_global) * 4;
                    const int32_t global_byte_idx = global_bit_idx / 8;
                    const int32_t sub_byte_offset = (global_bit_idx % 8) / 4;

                    const uint8_t packed_byte = weights_fp4[global_byte_idx];
                    const uint8_t raw_fp4 = (packed_byte >> (sub_byte_offset * 4)) & 0x0F;
                    uint8_t aligned_fp4x2 = (raw_fp4 << 4) | raw_fp4;
                    __half2_raw h2 = __nv_cvt_fp4x2_to_halfraw2(aligned_fp4x2, __NV_E2M1);
                    const float2 f2 = __half22float2(*reinterpret_cast<__half2 *>(&h2));
                    shmem_weights[n_local * tile_size_k + k_local] = __float2bfloat16(f2.x * scale);
                } else {
                    shmem_weights[n_local * tile_size_k + k_local] = __float2bfloat16(0.0f);
                }
            }
        }

        __syncthreads();

        const int32_t warp_m_base = (warp_id % 2) * 32;
        const int32_t warp_n_base = (warp_id / 2) * 32;

        for (int32_t k = 0; k < tile_size_k; k += 16) {
#pragma unroll
            for (int32_t m_step = 0; m_step < 2; ++m_step) {
                const int32_t warp_m = warp_m_base + m_step * 16;
                load_matrix_sync(a_frag, shmem_input + warp_m * tile_size_k + k, tile_size_k);

#pragma unroll
                for (int32_t n_step = 0; n_step < 2; ++n_step) {
                    const int32_t warp_n = warp_n_base + n_step * 16;
                    load_matrix_sync(b_frag, shmem_weights + warp_n * tile_size_k + k, tile_size_k);
                    mma_sync(c_frag[m_step][n_step], a_frag, b_frag, c_frag[m_step][n_step]);
                }
            }
        }

        __syncthreads();
    }

    const int32_t warp_m_base = (warp_id % 2) * 32;
    const int32_t warp_n_base = (warp_id / 2) * 32;

    auto shmem_out_buf = reinterpret_cast<float *>(dynamic_shmem);

#pragma unroll
    for (int32_t m_step = 0; m_step < 2; ++m_step) {
        const int32_t warp_m = warp_m_base + m_step * 16;
#pragma unroll
        for (int32_t n_step = 0; n_step < 2; ++n_step) {
            const int32_t warp_n = warp_n_base + n_step * 16;
            store_matrix_sync(shmem_out_buf + warp_m * tile_size_n + warp_n, c_frag[m_step][n_step], tile_size_n, mem_row_major);
        }
    }
    __syncthreads();

    for (int32_t i = tid; i < tile_size_m * tile_size_n; i += blockDim.x) {
        const int32_t m_local = i / tile_size_n;
        const int32_t n_local = i % tile_size_n;
        const int32_t m_global = block_m + m_local;
        const int32_t n_global = block_n + n_local;

        if (m_global < batch_num_tokens && n_global < local_output_dim) {
            const float bias_val = projection_bias != nullptr ? projection_bias[rank_offset + n_global] : 0.0f;
            float final_val = shmem_out_buf[m_local * tile_size_n + n_local] + bias_val;

            final_val = final_val / (1.0f + expf(-final_val));

            output_text_features[m_global * local_output_dim + n_global] = __float2bfloat16(final_val);
        }
    }
}

template __global__ void batched_projection_gemm_kernel<__nv_bfloat16>(
    const void ** __restrict__ device_table_A,
    const void ** __restrict__ device_table_B,
    void ** __restrict__ device_table_D,
    const cutlass::gemm::GemmCoord * __restrict__ device_shapes,
    const float * __restrict__ projection_bias,
    const float * __restrict__ quantization_scales,
    int32_t local_output_dim,
    int32_t input_feature_dim,
    int32_t rank_offset,
    int32_t tile_size_m,
    int32_t tile_size_n,
    int32_t tile_size_k
);

template __global__ void batched_projection_gemm_kernel<__nv_fp8_e4m3>(
    const void ** __restrict__ device_table_A,
    const void ** __restrict__ device_table_B,
    void ** __restrict__ device_table_D,
    const cutlass::gemm::GemmCoord * __restrict__ device_shapes,
    const float * __restrict__ projection_bias,
    const float * __restrict__ quantization_scales,
    int32_t local_output_dim,
    int32_t input_feature_dim,
    int32_t rank_offset,
    int32_t tile_size_m,
    int32_t tile_size_n,
    int32_t tile_size_k
);

template __global__ void batched_projection_gemm_kernel<__nv_fp4_e2m1>(
    const void ** __restrict__ device_table_A,
    const void ** __restrict__ device_table_B,
    void ** __restrict__ device_table_D,
    const cutlass::gemm::GemmCoord * __restrict__ device_shapes,
    const float * __restrict__ projection_bias,
    const float * __restrict__ quantization_scales,
    int32_t local_output_dim,
    int32_t input_feature_dim,
    int32_t rank_offset,
    int32_t tile_size_m,
    int32_t tile_size_n,
    int32_t tile_size_k
);
