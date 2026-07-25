#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>
#include "dequantize_fusion_mma_gguf_q4_k.cuh"

using namespace cute;

struct DequantQ4K {
    static __device__ __forceinline__ float dequantize_element(const BlockQ4K * __restrict__ input_B_quant, const int32_t global_n, const int32_t global_k, const int32_t K) {
        const int32_t total_weight_element_idx = global_n * K + global_k;
        const int32_t block_idx = total_weight_element_idx >> 8;
        const int32_t elem_in_block = total_weight_element_idx & 255;

        const BlockQ4K &block = input_B_quant[block_idx];

        const float d_val = __bfloat162float(block.d);
        const float dmin_val = __bfloat162float(block.dmin);

        const int32_t j = elem_in_block >> 6;
        const int32_t il = (elem_in_block >> 4) & 3;
        const int32_t pair_idx = elem_in_block & 15;

        const int32_t j_mod = j & 1;

        const uint8_t s_low = block.scales[j_mod * 4 + il];
        const uint8_t s_high = block.scales[8 + j_mod * 2 + (il >> 1)];

        const uint8_t sc = s_low & 63;
        const uint8_t min_sc = s_high & 63;

        const float d_super = d_val * static_cast<float>(sc);
        const float m_super = dmin_val * static_cast<float>(min_sc);

        const int32_t byte_idx = (j << 5) + (il << 2) + (pair_idx >> 1);
        const uint8_t packed_byte = block.qs[byte_idx];
        const uint8_t raw_q = (pair_idx & 1) == 0 ? packed_byte & 0x0F : packed_byte >> 4;

        return d_super * static_cast<float>(raw_q) - m_super;
    }
};

template<typename ElementAct, int TILE_M, int TILE_N, int TILE_K>
__global__ void fused_gemm_gguf_q4_k_kernel(
    ElementAct * __restrict__ output,
    const ElementAct * __restrict__ input_A,
    const BlockQ4K * __restrict__ input_B_quant,
    int32_t M, int32_t N, int32_t K
) {
    const int32_t block_m_coord = blockIdx.x * TILE_M;
    const int32_t block_n_coord = blockIdx.y * TILE_N;
    const int32_t tid = threadIdx.x;

    if (block_m_coord >= M || block_n_coord >= N) return;

    extern __shared__ uint8_t dynamic_shmem[];
    auto smem_A_ptr = reinterpret_cast<__nv_bfloat16 *>(dynamic_shmem);
    __nv_bfloat16 *smem_B_ptr = smem_A_ptr + TILE_M * TILE_K;

    auto smem_A_layout = make_layout(make_shape(Int<TILE_M>{}, Int<TILE_K>{}), LayoutRight{});
    auto smem_B_layout = make_layout(make_shape(Int<TILE_N>{}, Int<TILE_K>{}), LayoutRight{});

    auto smem_A_tensor = make_tensor(make_smem_ptr(smem_A_ptr), smem_A_layout);
    auto smem_B_tensor = make_tensor(make_smem_ptr(smem_B_ptr), smem_B_layout);

    auto mma_core = make_tiled_mma(
        SM80_16x8x16_F32BF16BF16F32_TN{},
        Layout<Shape<Int<2>, Int<2>, Int<1> > >{},
        Tile<Int<32>, Int<16>, Int<16> >{}
    );
    auto thr_mma = mma_core.get_slice(tid);
    auto tA_sA_partitioned = thr_mma.partition_A(smem_A_tensor);
    auto tB_sB_partitioned = thr_mma.partition_B(smem_B_tensor);
    auto smem_C_ptr = reinterpret_cast<float *>(dynamic_shmem);
    auto smem_C_layout = make_layout(make_shape(Int<TILE_M>{}, Int<TILE_N>{}), LayoutRight{});
    auto smem_C_tensor = make_tensor(make_smem_ptr(smem_C_ptr), smem_C_layout);
    auto tC_sC_partitioned = thr_mma.partition_C(smem_C_tensor);
    decltype(thr_mma.make_fragment_C(tC_sC_partitioned)) tC_rC;

#pragma unroll
    for (int32_t i = 0; i < size(tC_rC); ++i) {
        tC_rC(i) = 0.0f;
    }
    const int32_t num_k_tiles = (K + TILE_K - 1) / TILE_K;
    for (int32_t k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
#pragma unroll 4
        for (int32_t i = tid; i < TILE_M * TILE_K / 8; i += blockDim.x) {
            int32_t idx = i * 8;
            int32_t local_m = idx / TILE_K;
            int32_t local_k = idx % TILE_K;
            int32_t global_m = block_m_coord + local_m;
            int32_t global_k = k_tile * TILE_K + local_k;
            auto smem_ptr_u4 = reinterpret_cast<uint4 *>(&smem_A_ptr[local_m * TILE_K + local_k]);
            if (global_m < M && global_k < K) {
                if constexpr (std::is_same_v<ElementAct, bfloat16_t> || std::is_same_v<ElementAct, __nv_bfloat16>) {
                    *smem_ptr_u4 = *reinterpret_cast<const uint4 *>(&input_A[global_m * K + global_k]);
                } else if constexpr (std::is_same_v<ElementAct, __nv_fp8_e4m3>) {
                    auto base_bytes = reinterpret_cast<const uint8_t *>(input_A);
                    auto fp8_in = &base_bytes[global_m * K + global_k];
                    uint32_t u32_vals[4];
#pragma unroll
                    for (int v = 0; v < 4; ++v) {
                        uint16_t packed = (static_cast<uint16_t>(fp8_in[v * 2 + 1]) << 8) | fp8_in[v * 2];
                        __half2_raw h2 = __nv_cvt_fp8x2_to_halfraw2(packed, __NV_E4M3);
                        float2 f2 = __half22float2(*reinterpret_cast<__half2 *>(&h2));
                        __nv_bfloat16 bf_x = __float2bfloat16(f2.x);
                        __nv_bfloat16 bf_y = __float2bfloat16(f2.y);
                        uint16_t bits_x = *reinterpret_cast<uint16_t *>(&bf_x);
                        uint16_t bits_y = *reinterpret_cast<uint16_t *>(&bf_y);
                        u32_vals[v] = (static_cast<uint32_t>(bits_y) << 16) | bits_x;
                    }
                    *smem_ptr_u4 = make_uint4(u32_vals[0], u32_vals[1], u32_vals[2], u32_vals[3]);
                } else if constexpr (std::is_same_v<ElementAct, __nv_fp4_e2m1>) {
                    auto base_bytes = reinterpret_cast<const uint8_t *>(input_A);
                    int32_t global_fp4_element_idx = global_m * K + global_k;
                    auto fp4_in = &base_bytes[global_fp4_element_idx >> 1];
                    uint32_t u32_vals[4];

#pragma unroll
                    for (int v = 0; v < 4; ++v) {
                        uint8_t packed_byte = fp4_in[v];
                        __half2_raw h2 = __nv_cvt_fp4x2_to_halfraw2(packed_byte, __NV_E2M1);
                        __half2 *h2_ptr = reinterpret_cast<__half2 *>(&h2);
                        float2 f2 = __half22float2(*h2_ptr);

                        __nv_bfloat16 bf16_x = __float2bfloat16(f2.x);
                        __nv_bfloat16 bf16_y = __float2bfloat16(f2.y);

                        uint16_t bits_x = *reinterpret_cast<uint16_t *>(&bf16_x);
                        uint16_t bits_y = *reinterpret_cast<uint16_t *>(&bf16_y);
                        u32_vals[v] = (static_cast<uint32_t>(bits_y) << 16) | bits_x;

                        if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0 && k_tile == 0 && i == 0) {
                            printf("[A_LOAD TRACE] tid=%d | v=%d | raw_byte=0x%02X | cvt_h2_bits=0x%08X | f2=(%f, %f) | bf16_to_smem=(%f, %f)\n",
                                   tid, v, packed_byte, *reinterpret_cast<uint32_t *>(&h2), f2.x, f2.y, __bfloat162float(bf16_x), __bfloat162float(bf16_y));
                        }
                    }

                    uint4 final_u4 = make_uint4(u32_vals[0], u32_vals[1], u32_vals[2], u32_vals[3]);
                    *smem_ptr_u4 = final_u4;

                    if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0 && k_tile == 0 && i == 0) {
                        printf("[A_SMEM WRITE] tid=%d | written_u4=(0x%08X, 0x%08X, 0x%08X, 0x%08X)\n",
                               tid, final_u4.x, final_u4.y, final_u4.z, final_u4.w);
                    }
                }
            } else {
                *smem_ptr_u4 = make_uint4(0, 0, 0, 0);
            }
        }
#pragma unroll 2
        for (int32_t i = tid; i < TILE_N * TILE_K; i += blockDim.x) {
            int32_t local_n = i / TILE_K;
            int32_t local_k = i % TILE_K;
            int32_t global_n = block_n_coord + local_n;
            int32_t global_k = k_tile * TILE_K + local_k;

            if (global_n < N && global_k < K) {
                float out_fp32 = DequantQ4K::dequantize_element(input_B_quant, global_n, global_k, K);
                smem_B_ptr[local_n * TILE_K + local_k] = __float2bfloat16(out_fp32);

                if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0 && k_tile == 0 && i == 0) {
                    printf("[B_DEQUANT TRACE] FIRST ELEMENT ONLY | out_fp32=%f | bf16_bits=0x%04X\n",
                           out_fp32, *reinterpret_cast<uint16_t *>(&smem_B_ptr[local_n * TILE_K + local_k]));
                }
            } else {
                smem_B_ptr[local_n * TILE_K + local_k] = __float2bfloat16(0.0f);
            }
        }

        __syncthreads();

        auto tA_rA = thr_mma.make_fragment_A(tA_sA_partitioned);
        auto tB_rB = thr_mma.make_fragment_B(tB_sB_partitioned);

        cute::copy(tA_sA_partitioned, tA_rA);
        cute::copy(tB_sB_partitioned, tB_rB);

        if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0 && k_tile == 0) {
            printf("[MMA FRAGMENT PRE_GEMM] FIRST THREAD ONLY | size(tC_rC)=%d | fragment_C(0) before GEMM = %f\n",
                   static_cast<int>(size(tC_rC)), tC_rC(0));
        }

        gemm(mma_core, tA_rA, tB_rB, tC_rC);

        if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0 && k_tile == 0) {
            printf("[MMA FRAGMENT POST_GEMM] FIRST THREAD ONLY | fragment_C(0) after GEMM = %f\n", tC_rC(0));
        }

        __syncthreads();
    }

#pragma unroll
    for (int32_t i = 0; i < size(tC_rC); ++i) {
        tC_sC_partitioned(i) = tC_rC(i);
    }
    __syncthreads();

    if constexpr (std::is_same_v<ElementAct, bfloat16_t> || std::is_same_v<ElementAct, __nv_bfloat16>) {
        auto output_v2 = reinterpret_cast<__nv_bfloat162 *>(output);
        for (int32_t i = tid * 2; i < TILE_M * TILE_N; i += blockDim.x * 2) {
            int32_t m_local = i / TILE_N;
            int32_t n_local = i % TILE_N;
            int32_t global_m = block_m_coord + m_local;
            int32_t global_n = block_n_coord + n_local;
            if (global_m < M && global_n < N) {
                if (n_local + 1 < TILE_N && global_n + 1 < N) {
                    float val0 = smem_C_ptr[m_local * TILE_N + n_local];
                    float val1 = smem_C_ptr[m_local * TILE_N + n_local + 1];
                    int32_t target_idx = (global_m * N + global_n) >> 1;
                    output_v2[target_idx] = __halves2bfloat162(__float2bfloat16(val0), __float2bfloat16(val1));
                } else {
                    float val0 = smem_C_ptr[m_local * TILE_N + n_local];
                    output[global_m * N + global_n] = static_cast<ElementAct>(__float2bfloat16(val0));
                }
            }
        }
    } else if constexpr (std::is_same_v<ElementAct, __nv_fp8_e4m3>) {
        auto fp8_out = reinterpret_cast<uint8_t *>(output);
        for (int32_t i = tid; i < TILE_M * TILE_N; i += blockDim.x) {
            int32_t m_local = i / TILE_N;
            int32_t n_local = i % TILE_N;
            int32_t global_m = block_m_coord + m_local;
            int32_t global_n = block_n_coord + n_local;
            if (global_m < M && global_n < N) {
                float val = smem_C_ptr[m_local * TILE_N + n_local];
                __half h_val = __float2half(val);
                __half_raw h_raw = *reinterpret_cast<__half_raw *>(&h_val);
                fp8_out[global_m * N + global_n] = __nv_cvt_halfraw_to_fp8(h_raw, __NV_NOSAT, __NV_E4M3);
            }
        }
    } else if constexpr (std::is_same_v<ElementAct, __nv_fp4_e2m1>) {
        auto fp4_out = reinterpret_cast<uint8_t *>(output);
        for (int32_t i = tid; i < TILE_M * TILE_N / 2; i += blockDim.x) {
            int32_t total_elements_idx = i * 2;
            int32_t m_local = total_elements_idx / TILE_N;
            int32_t n_local = total_elements_idx % TILE_N;
            int32_t global_m = block_m_coord + m_local;
            int32_t global_n = block_n_coord + n_local;

            if (global_m < M && global_n < N) {
                float v0 = smem_C_ptr[m_local * TILE_N + n_local];
                float v1 = n_local + 1 < TILE_N && global_n + 1 < N ? smem_C_ptr[m_local * TILE_N + n_local + 1] : 0.0f;

                __half h0 = __float2half(v0);
                __half h1 = __float2half(v1);
                __half_raw h_raw0 = *reinterpret_cast<__half_raw *>(&h0);
                __half_raw h_raw1 = *reinterpret_cast<__half_raw *>(&h1);

                uint8_t r0 = __nv_cvt_halfraw_to_fp4(h_raw0, __NV_E2M1, cudaRoundNearest) & 0x0F;
                uint8_t r1 = __nv_cvt_halfraw_to_fp4(h_raw1, __NV_E2M1, cudaRoundNearest) & 0x0F;

                int32_t global_byte_row_stride = N >> 1;
                int32_t target_byte_idx = global_m * global_byte_row_stride + (global_n >> 1);

                fp4_out[target_byte_idx] = r0 | (r1 << 4);

                if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0 && i == 0) {
                    printf("[EPILOGUE TRACE WRITE] FIRST BYTE ONLY | target_byte_idx=%d | smem_v0=%f, smem_v1=%f | r0=0x%X, r1=0x%X | final_byte=0x%02X\n",
                           target_byte_idx, v0, v1, static_cast<int>(r0), static_cast<int>(r1), static_cast<int>(r0 | (r1 << 4)));
                }
            }
        }
    }
}

void run_fused_gemm_gguf_q4_k_bf16(bfloat16_t *output, const bfloat16_t *input_A, const BlockQ4K *input_B_quant, const int32_t M, const int32_t N, const int32_t K, dim3 grid, dim3 block, size_t shmem, cudaStream_t stream) {
    fused_gemm_gguf_q4_k_kernel<bfloat16_t, 64, 64, 32><<<grid, block, shmem, stream>>>(output, input_A, input_B_quant, M, N, K);
}

void run_fused_gemm_gguf_q4_k_fp8(__nv_fp8_e4m3 *output, const __nv_fp8_e4m3 *input_A, const BlockQ4K *input_B_quant, const int32_t M, const int32_t N, const int32_t K, dim3 grid, dim3 block, size_t shmem, cudaStream_t stream) {
    fused_gemm_gguf_q4_k_kernel<__nv_fp8_e4m3, 64, 64, 32><<<grid, block, shmem, stream>>>(output, input_A, input_B_quant, M, N, K);
}

void run_fused_gemm_gguf_q4_k_fp4(__nv_fp4_e2m1 *output, const __nv_fp4_e2m1 *input_A, const BlockQ4K *input_B_quant, const int32_t M, const int32_t N, const int32_t K, dim3 grid, dim3 block, size_t shmem, cudaStream_t stream) {
    fused_gemm_gguf_q4_k_kernel<__nv_fp4_e2m1, 64, 64, 32><<<grid, block, shmem, stream>>>(output, input_A, input_B_quant, M, N, K);
}
