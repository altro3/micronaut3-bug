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
                        auto h2_ptr = reinterpret_cast<__half2 *>(&h2);
                        float2 f2 = __half22float2(*h2_ptr);
                        __nv_bfloat16 bf16_x = __float2bfloat16(f2.x);
                        __nv_bfloat16 bf16_y = __float2bfloat16(f2.y);
                        uint16_t bits_x = *reinterpret_cast<uint16_t *>(&bf16_x);
                        uint16_t bits_y = *reinterpret_cast<uint16_t *>(&bf16_y);
                        u32_vals[v] = (static_cast<uint32_t>(bits_y) << 16) | bits_x;
                    }
                    *smem_ptr_u4 = make_uint4(u32_vals[0], u32_vals[1], u32_vals[2], u32_vals[3]);
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
            } else {
                smem_B_ptr[local_n * TILE_K + local_k] = __float2bfloat16(0.0f);
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
        uint8_t *smem_fp4 = dynamic_shmem;

        for (int32_t i = tid; i < TILE_M * TILE_N; i += blockDim.x) {
            int32_t m_local = i / TILE_N;
            int32_t n_local = i % TILE_N;
            float val = smem_C_ptr[m_local * TILE_N + n_local] / 2.0f;

            __half h_val = __float2half(val);
            __half_raw h_raw = *reinterpret_cast<__half_raw *>(&h_val);
            smem_fp4[i] = __nv_cvt_halfraw_to_fp4(h_raw, __NV_E2M1, cudaRoundNearest) & 0x0F;
        }
        __syncthreads();

        auto fp4_out = reinterpret_cast<uint32_t *>(output);
        int32_t total_u32_elements = TILE_M * TILE_N / 8;

        for (int32_t i = tid; i < total_u32_elements; i += blockDim.x) {
            int32_t local_u32_idx = i;
            int32_t base_elem_idx = local_u32_idx * 8;

            int32_t m_local = base_elem_idx / TILE_N;
            int32_t n_local = base_elem_idx % TILE_N;
            int32_t global_m = block_m_coord + m_local;
            int32_t global_n = block_n_coord + n_local;

            if (global_m < M && global_n < N) {
                uint32_t packed_val = 0;
#pragma unroll
                for (int v = 0; v < 8; ++v) {
                    packed_val |= static_cast<uint32_t>(smem_fp4[base_elem_idx + v]) << (v * 4);
                }

                int32_t global_element_idx = global_m * N + global_n;
                int32_t global_u32_idx = global_element_idx / 8;
                fp4_out[global_u32_idx] = packed_val;
            }
        }
    }
}

template<int TILE_M, int TILE_N, int TILE_K, int STAGES>
__global__ void __launch_bounds__(128, 1) fused_gemm_blackwell_q4_k_fp4_kernel(
    __nv_fp4_e2m1 * __restrict__ output,
    const __nv_fp4_e2m1 * __restrict__ input_A,
    const BlockQ4K * __restrict__ input_B_quant,
    const int32_t M, const int32_t N, const int32_t K
) {
    const int32_t tid = threadIdx.x;
    const int32_t block_m_coord = blockIdx.x * TILE_M;
    const int32_t block_n_coord = blockIdx.y * TILE_N;

    if (block_m_coord >= M || block_n_coord >= N) return;

    extern __shared__ uint8_t shared_storage[];

    auto smem_layout_A = composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<TILE_M>{}, Int<TILE_K / 2>{}), LayoutRight{})
    );

    constexpr int32_t blocks_per_tile_n = TILE_N / 256;
    auto smem_layout_B = make_layout(make_shape(Int<blocks_per_tile_n>{}, Int<TILE_K>{}), LayoutRight{});

    uint8_t *smem_A_ptr = shared_storage;
    BlockQ4K *smem_B_ptr = reinterpret_cast<BlockQ4K *>(smem_A_ptr + TILE_M * (TILE_K / 2) * STAGES);

    using TiledMma = decltype(make_tiled_mma(
        SM90_64x16x16_F32BF16BF16_RS{}
    ));
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);

    auto tC_gC = make_tensor(make_smem_ptr(static_cast<float *>(nullptr)), make_layout(make_shape(Int<TILE_M>{}, Int<TILE_N>{})));
    auto tC_rC = thr_mma.make_fragment_C(tC_gC);
    clear(tC_rC);

    const int32_t num_k_tiles = (K + TILE_K - 1) / TILE_K;

#pragma unroll 1
    for (int32_t k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        for (int32_t i = tid; i < TILE_M * (TILE_K / 2) / 16; i += blockDim.x) {
            const int32_t local_byte_idx = i * 16;
            const int32_t local_m = local_byte_idx / (TILE_K / 2);
            const int32_t local_k_byte = local_byte_idx % (TILE_K / 2);

            const int32_t global_m = block_m_coord + local_m;
            const int32_t global_k_byte = k_tile * TILE_K / 2 + local_k_byte;

            const auto src_ptr = reinterpret_cast<const uint4 *>(&input_A[global_m * (K / 2) + global_k_byte]);
            const auto dst_ptr = reinterpret_cast<uint4 *>(&smem_A_ptr[local_m * (TILE_K / 2) + local_k_byte]);

            if (global_m < M && global_k_byte < K / 2) {
                *dst_ptr = *src_ptr;
            } else {
                *dst_ptr = make_uint4(0, 0, 0, 0);
            }
        }

        for (int32_t i = tid; i < blocks_per_tile_n * TILE_K; i += blockDim.x) {
            const int32_t local_block_n = i / TILE_K;
            const int32_t local_k = i % TILE_K;

            const int32_t global_n = block_n_coord + local_block_n * 256;
            const int32_t global_weight_block_idx = global_n / 256 * (K / 256) + k_tile * TILE_K / 256 + local_k / 256;

            if (global_n < N) {
                smem_B_ptr[local_block_n * (TILE_K / 256) + local_k / 256] = input_B_quant[global_weight_block_idx];
            }
        }

        __syncthreads();

        auto tA_sA = thr_mma.partition_A(make_tensor(make_smem_ptr(smem_A_ptr), smem_layout_A));
        auto tA_rA = thr_mma.make_fragment_A(tA_sA);

        auto tB_layout_flat = make_layout(make_shape(Int<TILE_N>{}, Int<TILE_K>{}), LayoutRight{});
        auto tB_rB = thr_mma.make_fragment_B(make_tensor(make_smem_ptr(static_cast<__nv_bfloat16 *>(nullptr)), tB_layout_flat));

        cute::copy(tA_sA, tA_rA);

        auto tB_thr_tensor = thr_mma.partition_B(make_tensor(make_smem_ptr(static_cast<__nv_bfloat16 *>(nullptr)), tB_layout_flat));
        auto tB_thr_layout = tB_thr_tensor.layout();

#pragma unroll
#pragma unroll
        for (int32_t i = 0; i < size(tB_rB); ++i) {
            auto logical_coords = tB_thr_layout.get_flat_coord(i);
            int32_t logical_n = static_cast<int32_t>(cute::get<0>(logical_coords));
            int32_t logical_k = static_cast<int32_t>(cute::get<1>(logical_coords));

            int32_t global_k = k_tile * TILE_K + logical_k;

            int32_t super_block_idx = global_k / 128;
            int32_t block_idx = logical_n * (TILE_K / 128) + super_block_idx;
            int32_t elem_in_block = global_k % 128;

            const BlockQ4K &block = smem_B_ptr[block_idx];

            float d_val = __bfloat162float(block.d);
            float dmin_val = __bfloat162float(block.dmin);

            int32_t sub_block_idx = elem_in_block / 32;
            int32_t elem_idx = elem_in_block % 32;

            int32_t bit_offset_sc = sub_block_idx * 6;
            int32_t byte_offset_sc = bit_offset_sc / 8;
            int32_t bit_shift_sc = bit_offset_sc % 8;
            uint32_t val_sc = block.scales[byte_offset_sc] | (block.scales[byte_offset_sc + 1] << 8);
            if (byte_offset_sc + 2 < 12) {
                val_sc |= block.scales[byte_offset_sc + 2] << 16;
            }
            uint8_t sc = (val_sc >> bit_shift_sc) & 0x3F;

            int32_t bit_offset_min = (sub_block_idx + 4) * 6;
            int32_t byte_offset_min = bit_offset_min / 8;
            int32_t bit_shift_min = bit_offset_min % 8;
            uint32_t val_min = block.scales[byte_offset_min] | (block.scales[byte_offset_min + 1] << 8);
            if (byte_offset_min + 2 < 12) {
                val_min |= block.scales[byte_offset_min + 2] << 16;
            }
            uint8_t min_sc = (val_min >> bit_shift_min) & 0x3F;

            uint8_t qs_byte = block.qs[sub_block_idx * 16 + elem_idx % 16];
            uint8_t raw_q = elem_idx < 16 ? qs_byte & 0x0F : qs_byte >> 4;

            float out_fp32 = d_val * static_cast<float>(sc) * static_cast<float>(raw_q) - dmin_val * static_cast<float>(min_sc);
            tB_rB(i) = __float2bfloat16(out_fp32);
        }

        gemm(tiled_mma, tA_rA, tB_rB, tC_rC);

        __syncthreads();
    }

    auto smem_C_layout = make_layout(make_shape(Int<TILE_M>{}, Int<TILE_N>{}), LayoutRight{});
    auto smem_C_tensor = make_tensor(make_smem_ptr(reinterpret_cast<float *>(shared_storage)), smem_C_layout);
    auto tC_sC = thr_mma.partition_C(smem_C_tensor);

#pragma unroll
    for (int32_t i = 0; i < size(tC_rC); ++i) {
        tC_sC(i) = tC_rC(i);
    }
    __syncthreads();

    const int32_t total_packed_threads = TILE_M * TILE_N / 8;
    const auto fp4_global_out = reinterpret_cast<uint32_t *>(output);

    for (int32_t i = tid; i < total_packed_threads; i += blockDim.x) {
        const int32_t base_idx = i * 8;
        const int32_t local_m = base_idx / TILE_N;
        const int32_t local_n = base_idx % TILE_N;

        const int32_t global_m = block_m_coord + local_m;
        const int32_t global_n = block_n_coord + local_n;

        if (global_m < M && global_n < N) {
            uint32_t packed_val = 0;
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const float val = reinterpret_cast<float *>(shared_storage)[base_idx + v];

                __half h_val = __float2half(val / 2.0f);
                const __half_raw h_raw = *reinterpret_cast<__half_raw *>(&h_val);

                const uint32_t fp4_bits = __nv_cvt_halfraw_to_fp4(h_raw, __NV_E2M1, cudaRoundNearest) & 0x0F;
                packed_val |= fp4_bits << (v * 4);
            }

            const int32_t global_u32_idx = (global_m * (N / 2) + global_n) / 8;
            fp4_global_out[global_u32_idx] = packed_val;
        }
    }
}


void run_fused_gemm_gguf_q4_k_bf16(bfloat16_t *output, const bfloat16_t *input_A, const BlockQ4K *input_B_quant, const int32_t M, const int32_t N, const int32_t K, dim3 grid, dim3 block, size_t shmem, cudaStream_t stream) {
    fused_gemm_gguf_q4_k_kernel<bfloat16_t, 64, 64, 32><<<grid, block, shmem, stream>>>(output, input_A, input_B_quant, M, N, K);
}

void run_fused_gemm_gguf_q4_k_fp8(__nv_fp8_e4m3 *output, const __nv_fp8_e4m3 *input_A, const BlockQ4K *input_B_quant, const int32_t M, const int32_t N, const int32_t K, dim3 grid, dim3 block, size_t shmem, cudaStream_t stream) {
    fused_gemm_gguf_q4_k_kernel<__nv_fp8_e4m3, 64, 64, 32><<<grid, block, shmem, stream>>>(output, input_A, input_B_quant, M, N, K);
}

void run_fused_gemm_gguf_q4_k_fp4(
    __nv_fp4_e2m1 *output,
    const __nv_fp4_e2m1 *input_A,
    const BlockQ4K *input_B_quant,
    const int32_t M, const int32_t N, const int32_t K,
    dim3 grid, dim3 block, size_t shmem,
    cudaStream_t stream
) {
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 64;
    constexpr int TILE_K = 32;
    constexpr int STAGES = 2;

    constexpr size_t shmem_A = TILE_M * (TILE_K / 2) * STAGES;
    constexpr size_t shmem_B = TILE_N / 256 * TILE_K * sizeof(BlockQ4K);
    constexpr size_t shmem_C = TILE_M * TILE_N * sizeof(float);

    size_t total_shmem = std::max(shmem_A + shmem_B, shmem_C);

    if (total_shmem >= 48 * 1024) {
        cudaFuncSetAttribute(
            fused_gemm_blackwell_q4_k_fp4_kernel<TILE_M, TILE_N, TILE_K, STAGES>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            total_shmem
        );
    }

    fused_gemm_blackwell_q4_k_fp4_kernel<TILE_M, TILE_N, TILE_K, STAGES><<<grid, block, total_shmem, stream>>>(
        output, input_A, input_B_quant, M, N, K
    );
}
