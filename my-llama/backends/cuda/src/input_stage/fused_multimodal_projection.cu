#include "fused_multimodal_projection.cuh"
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

using namespace cute;

template <
    int TILE_M, int TILE_N, int TILE_K,
    typename T_Weight
>
__global__ void batched_projection_cutlass4_kernel(
    const void** __restrict__ device_table_A,
    const void** __restrict__ device_table_B,
    void** __restrict__ device_table_D,
    const cutlass::gemm::GemmCoord* __restrict__ device_shapes,
    const float* __restrict__ projection_bias,
    const float* __restrict__ quantization_scales,
    int32_t local_output_dim,
    int32_t input_feature_dim,
    int32_t rank_offset
) {
    const int32_t segment_id = blockIdx.z;
    cutlass::gemm::GemmCoord problem_size = device_shapes[segment_id];
    const int32_t batch_num_tokens = problem_size.m();

    if (batch_num_tokens <= 0) return;

    const int32_t block_m_coord = blockIdx.x * TILE_M;
    const int32_t block_n_coord = blockIdx.y * TILE_N;

    if (block_m_coord >= batch_num_tokens) return;

    auto input_ptr = static_cast<const bfloat16_t*>(device_table_A[segment_id]);
    const void* weight_ptr = device_table_B[segment_id];
    auto output_ptr = static_cast<bfloat16_t*>(device_table_D[segment_id]);

    extern __shared__ uint8_t dynamic_shmem[];
    auto smem_A_ptr = reinterpret_cast<bfloat16_t*>(dynamic_shmem);
    auto smem_B_ptr = smem_A_ptr + TILE_M * TILE_K;

    const int32_t tid = threadIdx.x;

    __shared__ float shared_scale;
    if (tid == 0) {
        shared_scale = (quantization_scales != nullptr) ? quantization_scales[segment_id] : 1.0f;
    }
    __syncthreads();
    float scale = shared_scale;

    auto gmem_A_layout = make_layout(make_shape(batch_num_tokens, input_feature_dim), LayoutRight{});
    auto smem_A_layout = make_layout(make_shape(Int<TILE_M>{}, Int<TILE_K>{}), LayoutRight{});
    auto smem_B_layout = make_layout(make_shape(Int<TILE_N>{}, Int<TILE_K>{}), LayoutRight{});

    auto gmem_A_tensor = make_tensor(make_gmem_ptr(input_ptr), gmem_A_layout);
    auto smem_A_tensor = make_tensor(make_smem_ptr(smem_A_ptr), smem_A_layout);
    auto smem_B_tensor = make_tensor(make_smem_ptr(smem_B_ptr), smem_B_layout);

    auto block_gmem_A = local_tile(gmem_A_tensor, make_shape(Int<TILE_M>{}, Int<TILE_K>{}), make_coord(blockIdx.x, _));
    auto tA_gmem_to_smem = local_partition(block_gmem_A, make_layout(make_shape(Int<128>{}), LayoutRight{}), tid);
    auto tA_smem = local_partition(smem_A_tensor, make_layout(make_shape(Int<128>{}), LayoutRight{}), tid);

    auto mma_core = TiledMMA<
        MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>,
        Layout<Shape<Int<2>, Int<2>, Int<1>>>,
        Tile<Int<32>, Int<16>, Int<16>>
    >{};
    auto thr_mma = mma_core.get_slice(tid);

    auto tC_gC = thr_mma.partition_C(make_tensor(make_gmem_ptr(output_ptr), make_layout(make_shape(batch_num_tokens, local_output_dim), LayoutRight{})));
    auto tC_rC = thr_mma.make_fragment_C(tC_gC);

    for (int i = 0; i < size(tC_rC); ++i) {
        tC_rC(i) = 0.0f;
    }

    auto tA_rA_view = thr_mma.partition_A(smem_A_tensor);
    auto tB_rB_view = thr_mma.partition_B(smem_B_tensor);

    int32_t num_k_tiles = (input_feature_dim + TILE_K - 1) / TILE_K;

    for (int32_t k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        #pragma unroll
        for (int i = 0; i < size<2>(tA_gmem_to_smem); ++i) {
            auto coord = tA_gmem_to_smem.coord(make_coord(_, _, i));
            int32_t global_m = block_m_coord + get<0>(coord);
            int32_t global_k = k_tile * TILE_K + get<1>(coord);

            if (global_m < batch_num_tokens && global_k < input_feature_dim) {
                copy(UniversalCopy<uint4>{}, tA_gmem_to_smem(_, _, i, k_tile), tA_smem(_, _, i));
            } else {
                #pragma unroll
                for (int v = 0; v < size<0>(tA_smem); ++v) {
                    tA_smem(v, _, i) = static_cast<bfloat16_t>(0.0f);
                }
            }
        }

        if constexpr (std::is_same_v<T_Weight, bfloat16_t>) {
            auto weights_bf16 = static_cast<const bfloat16_t*>(weight_ptr);
            auto gmem_B_layout = make_layout(make_shape(local_output_dim, input_feature_dim), LayoutRight{});
            auto gmem_B_tensor = make_tensor(make_gmem_ptr(weights_bf16), gmem_B_layout);
            auto block_gmem_B = local_tile(gmem_B_tensor, make_shape(Int<TILE_N>{}, Int<TILE_K>{}), make_coord(blockIdx.y, _));
            auto tB_gmem_to_smem = local_partition(block_gmem_B, make_layout(make_shape(Int<128>{}), LayoutRight{}), tid);
            auto tB_smem = local_partition(smem_B_tensor, make_layout(make_shape(Int<128>{}), LayoutRight{}), tid);

            #pragma unroll
            for (int i = 0; i < size<2>(tB_gmem_to_smem); ++i) {
                auto coord = tB_gmem_to_smem.coord(make_coord(_, _, i));
                int32_t global_n = block_n_coord + get<0>(coord);
                int32_t global_k = k_tile * TILE_K + get<1>(coord);

                if (global_n < local_output_dim && global_k < input_feature_dim) {
                    copy(UniversalCopy<uint4>{}, tB_gmem_to_smem(_, _, i, k_tile), tB_smem(_, _, i));
                } else {
                    #pragma unroll
                    for (int v = 0; v < size<0>(tB_smem); ++v) {
                        tB_smem(v, _, i) = static_cast<bfloat16_t>(0.0f);
                    }
                }
            }
        } else if constexpr (std::is_same_v<T_Weight, __nv_fp8_e4m3>) {
            auto weights_fp8 = static_cast<const uint8_t*>(weight_ptr);
            for (int i = tid; i < TILE_N * TILE_K; i += blockDim.x) {
                int32_t n_local = i / TILE_K;
                int32_t k_local = i % TILE_K;
                int32_t n_global = block_n_coord + n_local;
                int32_t k_global = k_tile * TILE_K + k_local;

                if (n_global < local_output_dim && k_global < input_feature_dim) {
                    uint8_t raw_fp8 = weights_fp8[n_global * input_feature_dim + k_global];
                    uint16_t packed_fp8x2 = (static_cast<uint16_t>(raw_fp8) << 8) | raw_fp8;
                    __half2_raw h2 = __nv_cvt_fp8x2_to_halfraw2(packed_fp8x2, __NV_E4M3);
                    float2 f2 = __half22float2(*reinterpret_cast<__half2*>(&h2));
                    smem_B_ptr[n_local * TILE_K + k_local] = static_cast<bfloat16_t>(f2.x * scale);
                } else {
                    smem_B_ptr[n_local * TILE_K + k_local] = static_cast<bfloat16_t>(0.0f);
                }
            }
        } else if constexpr (std::is_same_v<T_Weight, __nv_fp4_e2m1>) {
            auto weights_fp4 = static_cast<const uint8_t*>(weight_ptr);
            for (int i = tid; i < TILE_N * TILE_K; i += blockDim.x) {
                int32_t n_local = i / TILE_K;
                int32_t k_local = i % TILE_K;
                int32_t n_global = block_n_coord + n_local;
                int32_t k_global = k_tile * TILE_K + k_local;

                if (n_global < local_output_dim && k_global < input_feature_dim) {
                    int32_t global_bit_idx = (n_global * input_feature_dim + k_global) * 4;
                    int32_t global_byte_idx = global_bit_idx / 8;
                    int32_t sub_byte_offset = (global_bit_idx % 8) / 4;

                    uint8_t packed_byte = weights_fp4[global_byte_idx];
                    uint8_t raw_fp4 = (packed_byte >> (sub_byte_offset * 4)) & 0x0F;
                    uint8_t aligned_fp4x2 = (raw_fp4 << 4) | raw_fp4;
                    __half2_raw h2 = __nv_cvt_fp4x2_to_halfraw2(aligned_fp4x2, __NV_E2M1);
                    float2 f2 = __half22float2(*reinterpret_cast<__half2*>(&h2));
                    smem_B_ptr[n_local * TILE_K + k_local] = static_cast<bfloat16_t>(f2.x * scale);
                } else {
                    smem_B_ptr[n_local * TILE_K + k_local] = static_cast<bfloat16_t>(0.0f);
                }
            }
        }

        cp_async_wait<0>();
        __syncthreads();

        auto tA_rA = thr_mma.make_fragment_A(tA_rA_view);
        auto tB_rB = thr_mma.make_fragment_B(tB_rB_view);

        gemm(mma_core, tA_rA, tB_rB, tC_rC);

        __syncthreads();
    }

    auto smem_out_buf = reinterpret_cast<float*>(dynamic_shmem);
    auto tC_sC_tensor = make_tensor(make_smem_ptr(smem_out_buf), make_layout(make_shape(Int<TILE_M>{}, Int<TILE_N>{}), LayoutRight{}));
    auto tC_sC = thr_mma.partition_C(tC_sC_tensor);

    #pragma unroll
    for (int i = 0; i < size(tC_rC); ++i) {
        tC_sC(i) = tC_rC(i);
    }
    __syncthreads();

    for (int32_t i = tid; i < TILE_M * TILE_N; i += blockDim.x) {
        int32_t m_local = i / TILE_N;
        int32_t n_local = i % TILE_N;
        int32_t m_global = block_m_coord + m_local;
        int32_t n_global = block_n_coord + n_local;

        if (m_global < batch_num_tokens && n_global < local_output_dim) {
            float bias_val = (projection_bias != nullptr) ? projection_bias[rank_offset + n_global] : 0.0f;
            float final_val = smem_out_buf[m_local * TILE_N + n_local] + bias_val;

            final_val = final_val / (1.0f + __expf(-final_val));

            output_ptr[m_global * local_output_dim + n_global] = static_cast<bfloat16_t>(final_val);
        }
    }
}

template __global__ void batched_projection_cutlass4_kernel<64, 64, 32, bfloat16_t>(
    const void** __restrict__ device_table_A, const void** __restrict__ device_table_B, void** __restrict__ device_table_D,
    const cutlass::gemm::GemmCoord* __restrict__ device_shapes, const float* __restrict__ projection_bias, const float* __restrict__ quantization_scales,
    int32_t local_output_dim, int32_t input_feature_dim, int32_t rank_offset
);

template __global__ void batched_projection_cutlass4_kernel<64, 64, 32, __nv_fp8_e4m3>(
    const void ** __restrict__ device_table_A, const void ** __restrict__ device_table_B, void ** __restrict__ device_table_D,
    const cutlass::gemm::GemmCoord * __restrict__ device_shapes, const float * __restrict__ projection_bias, const float * __restrict__ quantization_scales,
    int32_t local_output_dim, int32_t input_feature_dim, int32_t rank_offset
);

template __global__ void batched_projection_cutlass4_kernel<64, 64, 32, __nv_fp4_e2m1>(
    const void ** __restrict__ device_table_A, const void ** __restrict__ device_table_B, void ** __restrict__ device_table_D,
    const cutlass::gemm::GemmCoord * __restrict__ device_shapes, const float * __restrict__ projection_bias, const float * __restrict__ quantization_scales,
    int32_t local_output_dim, int32_t input_feature_dim, int32_t rank_offset
);
