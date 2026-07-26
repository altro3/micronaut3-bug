#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>
#include "dequantize_fusion_mma_gguf_q4_k.cuh"

using namespace cute;

struct DequantQ4K {
    static __device__ __forceinline__ float dequantize_element(const BlockQ4K * __restrict__ input_B_quant, const int32_t global_n, const int32_t global_k, const int32_t K) {
        const int32_t super_block_idx = global_k / 256;
        const int32_t block_idx = global_n * (K / 256) + super_block_idx;
        const int32_t elem_in_block = global_k % 256;

        const BlockQ4K &block = input_B_quant[block_idx];

        const float d_val = __bfloat162float(block.d);
        const float dmin_val = __bfloat162float(block.dmin);

        const int32_t sub_block_idx = elem_in_block / 32;
        const int32_t elem_idx = elem_in_block % 32;

        const int32_t bit_offset_sc = sub_block_idx * 6;
        const int32_t byte_offset_sc = bit_offset_sc / 8;
        const int32_t bit_shift_sc = bit_offset_sc % 8;
        uint32_t val_sc = block.scales[byte_offset_sc] | (block.scales[byte_offset_sc + 1] << 8);
        if (byte_offset_sc + 2 < 12) {
            val_sc |= block.scales[byte_offset_sc + 2] << 16;
        }
        const uint8_t sc = (val_sc >> bit_shift_sc) & 0x3F;

        const int32_t bit_offset_min = (sub_block_idx + 8) * 6;
        const int32_t byte_offset_min = bit_offset_min / 8;
        const int32_t bit_shift_min = bit_offset_min % 8;
        uint32_t val_min = block.scales[byte_offset_min] | (block.scales[byte_offset_min + 1] << 8);
        if (byte_offset_min + 2 < 12) {
            val_min |= block.scales[byte_offset_min + 2] << 16;
        }
        const uint8_t min_sc = (val_min >> bit_shift_min) & 0x3F;

        const uint8_t qs_byte = block.qs[sub_block_idx * 16 + elem_idx % 16];
        const uint8_t raw_q = elem_idx < 16 ? qs_byte & 0x0F : qs_byte >> 4;

        return d_val * static_cast<float>(sc) * static_cast<float>(raw_q) - dmin_val * static_cast<float>(min_sc);
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
            int32_t idx = i << 3;
            int32_t local_m = idx >> 5;
            int32_t local_k = idx & 31;
            int32_t global_m = block_m_coord + local_m;
            int32_t global_k = k_tile * TILE_K + local_k;
            auto smem_ptr_u4 = reinterpret_cast<uint4 *>(&smem_A_ptr[(local_m << 5) + local_k]);

            if (global_m < M && global_k < K) {
                int32_t row_offset = global_m * K + global_k;

                if constexpr (std::is_same_v<ElementAct, bfloat16_t> || std::is_same_v<ElementAct, __nv_bfloat16>) {
                    *smem_ptr_u4 = *reinterpret_cast<const uint4 *>(&input_A[global_m * K + global_k]);
                } else if constexpr (std::is_same_v<ElementAct, __nv_fp8_e4m3>) {
                    auto fp8_in_ptr = reinterpret_cast<const uint2 *>(reinterpret_cast<const uint8_t *>(input_A) + row_offset);

                    uint2 packed_fp8_val = *fp8_in_ptr;
                    uint32_t u32_vals[4];

                    uint16_t packed0 = packed_fp8_val.x & 0xFFFF;
                    __half2_raw h2_0 = __nv_cvt_fp8x2_to_halfraw2(packed0, __NV_E4M3);
                    float2 f2_0 = __half22float2(*reinterpret_cast<__half2 *>(&h2_0));
                    u32_vals[0] = (static_cast<uint32_t>(*reinterpret_cast<uint16_t *>(&__float2bfloat16(f2_0.y))) << 16) | *reinterpret_cast<uint16_t *>(&__float2bfloat16(f2_0.x));

                    uint16_t packed1 = packed_fp8_val.x >> 16;
                    __half2_raw h2_1 = __nv_cvt_fp8x2_to_halfraw2(packed1, __NV_E4M3);
                    float2 f2_1 = __half22float2(*reinterpret_cast<__half2 *>(&h2_1));
                    u32_vals[1] = (static_cast<uint32_t>(*reinterpret_cast<uint16_t *>(&__float2bfloat16(f2_1.y))) << 16) | *reinterpret_cast<uint16_t *>(&__float2bfloat16(f2_1.x));

                    uint16_t packed2 = packed_fp8_val.y & 0xFFFF;
                    __half2_raw h2_2 = __nv_cvt_fp8x2_to_halfraw2(packed2, __NV_E4M3);
                    float2 f2_2 = __half22float2(*reinterpret_cast<__half2 *>(&h2_2));
                    u32_vals[2] = (static_cast<uint32_t>(*reinterpret_cast<uint16_t *>(&__float2bfloat16(f2_2.y))) << 16) | *reinterpret_cast<uint16_t *>(&__float2bfloat16(f2_2.x));

                    uint16_t packed3 = packed_fp8_val.y >> 16;
                    __half2_raw h2_3 = __nv_cvt_fp8x2_to_halfraw2(packed3, __NV_E4M3);
                    float2 f2_3 = __half22float2(*reinterpret_cast<__half2 *>(&h2_3));
                    u32_vals[3] = (static_cast<uint32_t>(*reinterpret_cast<uint16_t *>(&__float2bfloat16(f2_3.y))) << 16) | *reinterpret_cast<uint16_t *>(&__float2bfloat16(f2_3.x));

                    *smem_ptr_u4 = make_uint4(u32_vals[0], u32_vals[1], u32_vals[2], u32_vals[3]);
                } else if constexpr (std::is_same_v<ElementAct, __nv_fp4_e2m1>) {
                    auto fp4_in = reinterpret_cast<const uint8_t *>(input_A) + ((global_m * K + global_k) >> 1);
                    uint32_t u32_vals[4];
#pragma unroll
                    for (int v = 0; v < 4; ++v) {
                        uint8_t packed_byte = fp4_in[v];
                        __half2_raw h2 = __nv_cvt_fp4x2_to_halfraw2(packed_byte, __NV_E2M1);
                        float2 f2 = __half22float2(*reinterpret_cast<__half2 *>(&h2));
                        __nv_bfloat16 bf16_x = __float2bfloat16(f2.x);
                        __nv_bfloat16 bf16_y = __float2bfloat16(f2.y);
                        u32_vals[v] = (static_cast<uint32_t>(*reinterpret_cast<uint16_t *>(&bf16_y)) << 16) | *reinterpret_cast<uint16_t *>(&bf16_x);
                    }
                    *smem_ptr_u4 = make_uint4(u32_vals[0], u32_vals[1], u32_vals[2], u32_vals[3]);
                }
            } else {
                *smem_ptr_u4 = make_uint4(0, 0, 0, 0);
            }
        }

#pragma unroll 2
        for (int32_t i = tid; i < TILE_N * TILE_K; i += blockDim.x) {
            int32_t local_n = i >> 5;
            int32_t local_k = i & 31;
            int32_t global_n = block_n_coord + local_n;
            int32_t global_k = k_tile * TILE_K + local_k;

            if (global_n < N && global_k < K) {
                float out_fp32 = DequantQ4K::dequantize_element(input_B_quant, global_n, global_k, K);
                smem_B_ptr[(local_n << 5) + local_k] = __float2bfloat16(out_fp32);
            } else {
                smem_B_ptr[(local_n << 5) + local_k] = __float2bfloat16(0.0f);
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
        auto fp4_out = reinterpret_cast<uint32_t *>(output);
        constexpr float fp4_scale = 2.0f;
        for (int32_t i = tid; i < TILE_M * TILE_N; i += blockDim.x) {
            int32_t m_local = i / TILE_N;
            int32_t n_local = i % TILE_N;
            int32_t global_m = block_m_coord + m_local;
            int32_t global_n = block_n_coord + n_local;

            if (global_m < M && global_n < N) {
                float val = smem_C_ptr[m_local * TILE_N + n_local] / fp4_scale;

                __half h_val = __float2half(val);
                __half_raw h_raw = *reinterpret_cast<__half_raw *>(&h_val);

                uint8_t res_fp4 = __nv_cvt_halfraw_to_fp4(h_raw, __NV_E2M1, cudaRoundNearest) & 0x0F;

                int32_t global_element_idx = global_m * N + global_n;
                int32_t global_u32_idx = global_element_idx / 8;
                int32_t shift = global_element_idx % 8 * 4;

                uint32_t mask = ~(0x0F << shift);
                uint32_t value_to_write = static_cast<uint32_t>(res_fp4) << shift;

                uint32_t *target_ptr = &fp4_out[global_u32_idx];
                atomicAnd(target_ptr, mask);
                atomicOr(target_ptr, value_to_write);
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
