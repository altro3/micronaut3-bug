#include "fused_multimodal_projection.cuh"
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

using namespace cute;

template<int TILE_M, int TILE_N, int TILE_K, typename T_Weight>
__global__ void batched_projection_cutlass4_kernel(
    ProjectionParams params,
    const float * __restrict__ projection_bias,
    const float * __restrict__ quantization_scales,
    int32_t local_output_dim,
    int32_t input_feature_dim,
    int32_t rank_offset
) {
    const int32_t segment_id = blockIdx.z;
    cutlass::gemm::GemmCoord problem_size = params.shapes[segment_id];
    const int32_t batch_num_tokens = problem_size.m();

    if (batch_num_tokens <= 0) return;

    const int32_t block_m_coord = blockIdx.x * TILE_M;
    const int32_t block_n_coord = blockIdx.y * TILE_N;

    if (block_m_coord >= batch_num_tokens) return;

    auto input_ptr = static_cast<const bfloat16_t *>(params.inputs[segment_id]);
    const void *weight_ptr = params.weights[segment_id];
    auto output_ptr = static_cast<bfloat16_t *>(params.outputs[segment_id]);

    extern __shared__ uint8_t dynamic_shmem[];
    auto smem_A_ptr = reinterpret_cast<bfloat16_t *>(dynamic_shmem);
    auto smem_B_ptr = smem_A_ptr + TILE_M * TILE_K;

    const int32_t tid = threadIdx.x;

    __shared__ float shared_scale;
    if (tid == 0) {
        shared_scale = quantization_scales != nullptr ? quantization_scales[segment_id] : 1.0f;
    }
    __syncthreads();
    float scale = shared_scale;

    auto smem_A_layout = make_layout(make_shape(Int<TILE_M>{}, Int<TILE_K>{}), LayoutRight{});
    auto smem_B_layout = make_layout(make_shape(Int<TILE_N>{}, Int<TILE_K>{}), LayoutLeft{});

    auto smem_A_tensor = make_tensor(make_smem_ptr(smem_A_ptr), smem_A_layout);
    auto smem_B_tensor = make_tensor(make_smem_ptr(smem_B_ptr), smem_B_layout);

    auto mma_core = make_tiled_mma(
        SM80_16x8x16_F32BF16BF16F32_TN{},
        Layout<Shape<Int<2>, Int<2>, Int<1> > >{},
        Tile<Int<32>, Int<16>, Int<16> >{}
    );
    auto thr_mma = mma_core.get_slice(tid);

    auto smem_C_ptr = reinterpret_cast<float *>(dynamic_shmem);
    auto smem_C_layout = make_layout(make_shape(Int<TILE_M>{}, Int<TILE_N>{}), LayoutRight{});
    auto smem_C_tensor = make_tensor(make_smem_ptr(smem_C_ptr), smem_C_layout);
    auto tC_sC_partitioned = thr_mma.partition_C(smem_C_tensor);

    decltype(thr_mma.make_fragment_C(tC_sC_partitioned)) tC_rC;

#pragma unroll
    for (int32_t i = 0; i < size(tC_rC); ++i) {
        tC_rC(i) = 0.0f;
    }

    auto tA_sA_partitioned = thr_mma.partition_A(smem_A_tensor);
    auto tB_sB_partitioned = thr_mma.partition_B(smem_B_tensor);

    int32_t num_k_tiles = (input_feature_dim + TILE_K - 1) / TILE_K;

    for (int32_t k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
#pragma unroll 4
        for (int32_t i = tid; i < TILE_M * TILE_K / 8; i += blockDim.x) {
            int32_t idx = i * 8;
            int32_t local_m = idx / TILE_K;
            int32_t local_k = idx % TILE_K;
            int32_t global_m = block_m_coord + local_m;
            int32_t global_k = k_tile * TILE_K + local_k;

            auto smem_ptr_u4 = reinterpret_cast<uint4 *>(&smem_A_ptr[local_m * TILE_K + local_k]);
            if (global_m < batch_num_tokens && global_k < input_feature_dim) {
                *smem_ptr_u4 = *reinterpret_cast<const uint4 *>(&input_ptr[global_m * input_feature_dim + global_k]);
            } else {
                *smem_ptr_u4 = make_uint4(0, 0, 0, 0);
            }
        }
        if constexpr (std::is_same_v<T_Weight, bfloat16_t>) {
            auto weights_bf16 = static_cast<const bfloat16_t *>(weight_ptr);
#pragma unroll 4
            for (int32_t i = tid; i < TILE_N * TILE_K / 8; i += blockDim.x) {
                int32_t idx = i * 8;
                int32_t local_n = idx / TILE_K;
                int32_t local_k = idx % TILE_K;
                int32_t global_n = block_n_coord + local_n;
                int32_t global_k_weight = k_tile * TILE_K + local_k;

                if (global_n < local_output_dim && global_k_weight < input_feature_dim) {
                    uint4 data = *reinterpret_cast<const uint4 *>(&weights_bf16[global_n * input_feature_dim + global_k_weight]);
                    auto reg_data = reinterpret_cast<bfloat16_t *>(&data);
#pragma unroll
                    for (int v = 0; v < 8; ++v) {
                        smem_B_ptr[(local_k + v) * TILE_N + local_n] = reg_data[v];
                    }
                } else {
#pragma unroll
                    for (int v = 0; v < 8; ++v) {
                        smem_B_ptr[(local_k + v) * TILE_N + local_n] = static_cast<bfloat16_t>(0.0f);
                    }
                }
            }
        } else if constexpr (std::is_same_v<T_Weight, __nv_fp8_e4m3>) {
            auto weights_fp8 = static_cast<const uint8_t *>(weight_ptr);
#pragma unroll 4
            for (int32_t i = tid; i < (TILE_N * TILE_K) / 16; i += blockDim.x) {
                int32_t idx = i * 16;
                int32_t local_n = idx / TILE_K;
                int32_t local_k = idx % TILE_K;
                int32_t global_n = block_n_coord + local_n;
                int32_t global_k_weight = k_tile * TILE_K + local_k;

                if (global_n < local_output_dim && global_k_weight < input_feature_dim) {
                    uint4 packed_v = *reinterpret_cast<const uint4 *>(&weights_fp8[global_n * input_feature_dim + global_k_weight]);
                    uint8_t *raw_bytes = reinterpret_cast<uint8_t *>(&packed_v);

#pragma unroll
                    for (int v = 0; v < 16; ++v) {
                        uint8_t raw_fp8 = raw_bytes[v];
                        uint16_t packed_fp8x2 = (static_cast<uint16_t>(raw_fp8) << 8) | raw_fp8;
                        __half2_raw h2 = __nv_cvt_fp8x2_to_halfraw2(packed_fp8x2, __NV_E4M3);
                        float2 f2 = __half22float2(*reinterpret_cast<__half2 *>(&h2));
                        int32_t curr_k = local_k + v;
                        smem_B_ptr[curr_k * TILE_N + local_n] = static_cast<bfloat16_t>(f2.x * scale);
                    }
                } else {
#pragma unroll
                    for (int v = 0; v < 16; ++v) {
                        int32_t curr_k = local_k + v;
                        smem_B_ptr[curr_k * TILE_N + local_n] = static_cast<bfloat16_t>(0.0f);
                    }
                }
            }
        } else if constexpr (std::is_same_v<T_Weight, __nv_fp4_e2m1>) {
            auto weights_fp4 = static_cast<const uint8_t *>(weight_ptr);
#pragma unroll 4
            for (int32_t i = tid; i < (TILE_N * TILE_K) / 32; i += blockDim.x) {
                int32_t idx = i * 32;
                int32_t local_n = idx / TILE_K;
                int32_t local_k = idx % TILE_K;
                int32_t global_n = block_n_coord + local_n;
                int32_t global_k_weight = k_tile * TILE_K + local_k;

                if (global_n < local_output_dim && global_k_weight < input_feature_dim) {
                    int32_t byte_idx = (global_n * input_feature_dim + global_k_weight) / 2;
                    uint4 packed_v = *reinterpret_cast<const uint4 *>(&weights_fp4[byte_idx]);
                    uint8_t *raw_bytes = reinterpret_cast<uint8_t *>(&packed_v);

#pragma unroll
                    for (int v = 0; v < 16; ++v) {
                        uint8_t byte_val = raw_bytes[v];
                        uint8_t raw_fp4_low = byte_val & 0x0F;
                        uint8_t raw_fp4_high = (byte_val >> 4) & 0x0F;

                        uint8_t aligned_low = (raw_fp4_low << 4) | raw_fp4_low;
                        __half2_raw h2_l = __nv_cvt_fp4x2_to_halfraw2(aligned_low, __NV_E2M1);
                        float2 f2_l = __half22float2(*reinterpret_cast<__half2 *>(&h2_l));
                        smem_B_ptr[(local_k + v * 2) * TILE_N + local_n] = static_cast<bfloat16_t>(f2_l.x * scale);

                        uint8_t aligned_high = (raw_fp4_high << 4) | raw_fp4_high;
                        __half2_raw h2_h = __nv_cvt_fp4x2_to_halfraw2(aligned_high, __NV_E2M1);
                        float2 f2_h = __half22float2(*reinterpret_cast<__half2 *>(&h2_h));
                        smem_B_ptr[(local_k + v * 2 + 1) * TILE_N + local_n] = static_cast<bfloat16_t>(f2_h.x * scale);
                    }
                } else {
#pragma unroll
                    for (int v = 0; v < 32; ++v) {
                        smem_B_ptr[(local_k + v) * TILE_N + local_n] = static_cast<bfloat16_t>(0.0f);
                    }
                }
            }
        }

        __syncthreads();

        auto tA_rA = thr_mma.make_fragment_A(tA_sA_partitioned);
        auto tB_rB = thr_mma.make_fragment_B(tB_sB_partitioned);

        cute::copy(tA_sA_partitioned, tA_rA);
        cute::copy(tB_sB_partitioned, tB_rB);

        gemm(mma_core, tA_rA, tB_rB, tC_rC);

        __syncthreads();
    }

#pragma unroll
    for (int32_t i = 0; i < size(tC_rC); ++i) {
        tC_sC_partitioned(i) = tC_rC(i);
    }
    __syncthreads();

    auto output_v2 = reinterpret_cast<__nv_bfloat162 *>(output_ptr);

    for (int32_t i = tid * 2; i < TILE_M * TILE_N; i += blockDim.x * 2) {
        int32_t m_local = i / TILE_N;
        int32_t n_local = i % TILE_N;
        int32_t global_m = block_m_coord + m_local;
        int32_t global_n = block_n_coord + n_local;

        if (global_m < batch_num_tokens && global_n + 1 < local_output_dim) {
            float bias0 = projection_bias != nullptr ? projection_bias[rank_offset + global_n] : 0.0f;
            float bias1 = projection_bias != nullptr ? projection_bias[rank_offset + global_n + 1] : 0.0f;

            float val0 = smem_C_ptr[m_local * TILE_N + n_local] + bias0;
            float val1 = smem_C_ptr[m_local * TILE_N + n_local + 1] + bias1;

            val0 = val0 / (1.0f + __expf(-val0));
            val1 = val1 / (1.0f + __expf(-val1));

            __nv_bfloat16 out0 = __float2bfloat16(val0);
            __nv_bfloat16 out1 = __float2bfloat16(val1);

            int32_t target_idx = (global_m * local_output_dim + global_n) >> 1;
            output_v2[target_idx] = __halves2bfloat162(out0, out1);
        } else if (global_m < batch_num_tokens && global_n < local_output_dim) {
            float bias0 = projection_bias != nullptr ? projection_bias[rank_offset + global_n] : 0.0f;
            float val0 = smem_C_ptr[m_local * TILE_N + n_local] + bias0;
            val0 = val0 / (1.0f + __expf(-val0));
            output_ptr[global_m * local_output_dim + global_n] = bfloat16_t(val0);
        }
    }
}

void run_batched_projection_bf16(ProjectionParams params, const float *projection_bias, const float *quantization_scales, int32_t local_output_dim, int32_t input_feature_dim, int32_t rank_offset, dim3 grid, dim3 block, size_t shmem, cudaStream_t stream) {
    batched_projection_cutlass4_kernel<64, 64, 32, bfloat16_t><<<grid, block, shmem, stream>>>(params, projection_bias, quantization_scales, local_output_dim, input_feature_dim, rank_offset);
}

void run_batched_projection_fp8(ProjectionParams params, const float *projection_bias, const float *quantization_scales, int32_t local_output_dim, int32_t input_feature_dim, int32_t rank_offset, dim3 grid, dim3 block, size_t shmem, cudaStream_t stream) {
    batched_projection_cutlass4_kernel<64, 64, 32, __nv_fp8_e4m3><<<grid, block, shmem, stream>>>(params, projection_bias, quantization_scales, local_output_dim, input_feature_dim, rank_offset);
}

void run_batched_projection_fp4(ProjectionParams params, const float *projection_bias, const float *quantization_scales, int32_t local_output_dim, int32_t input_feature_dim, int32_t rank_offset, dim3 grid, dim3 block, size_t shmem, cudaStream_t stream) {
    batched_projection_cutlass4_kernel<64, 64, 32, __nv_fp4_e2m1><<<grid, block, shmem, stream>>>(params, projection_bias, quantization_scales, local_output_dim, input_feature_dim, rank_offset);
}
