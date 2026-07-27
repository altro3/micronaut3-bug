#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda/barrier>
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

#include "dequantize_fusion_mma_gguf_q4_k.cuh"

using namespace cute;

__device__ __forceinline__ void dequantize_q4_k_to_fp4(
    const BlockQ4K &block,
    const int local_k_idx,
    float_e2m1_t *out_fp4) {
    const int sub_block_idx = local_k_idx / 32;
    const int element_idx = local_k_idx % 32;

    const float d_val = __bfloat162float(block.d);
    const float dmin_val = __bfloat162float(block.dmin);

    const uint8_t sc_byte = block.scales[sub_block_idx * 2 + element_idx / 16];
    const float scale = element_idx % 16 < 8 ? sc_byte & 0x0F : sc_byte >> 4;

    const uint8_t q_byte = block.qs[(sub_block_idx * 32 + element_idx) / 2];
    const uint8_t q_raw = element_idx % 2 == 0 ? q_byte & 0x0F : q_byte >> 4;

    const float dequantized_f32 = d_val * scale * q_raw - dmin_val;

    *out_fp4 = float_e2m1_t(dequantized_f32);
}

template<
    class TileM, class TileN, class TileK,
    class ElementA, class ElementC, class StageCount
>
__global__ void __launch_bounds__(128, 2)
dequantize_fusion_mma_gguf_q4_k_kernel(
    const ElementA *ptr_A, const int stride_A,
    const BlockQ4K *ptr_B_q4,
    ElementC *ptr_C, const int stride_C,
    const int M, const int N, const int K) {
    using namespace cute;

    extern __shared__ uint8_t shared_storage[];

    const int block_m = blockIdx.x;
    const int block_n = blockIdx.y;
    int tid = threadIdx.x;

    const auto tShapeM = make_shape(M);
    const auto tShapeN = make_shape(N);
    const auto tShapeK = make_shape(K);

    Tensor gA = make_tensor(make_gmem_ptr(ptr_A), make_layout(make_shape(tShapeM, tShapeK), make_stride(stride_A, _1{})));
    Tensor gC = make_tensor(make_gmem_ptr(ptr_C), make_layout(make_shape(tShapeM, tShapeN), make_stride(stride_C, _1{})));

    Tensor lA = local_tile(gA, make_tile(TileM{}, TileK{}), make_coord(block_m, _));
    Tensor lC = local_tile(gC, make_tile(TileM{}, TileN{}), make_coord(block_m, block_n));

    const int blocks_per_tile_k = TileK::value / 256;

    auto smem_layout_A = make_layout(make_shape(TileM{}, TileK{}, StageCount{}), LayoutRight{});

    Tensor sA = make_tensor(make_smem_ptr(static_cast<ElementA *>(shared_storage)), smem_layout_A);
    const auto sB_q4 = static_cast<BlockQ4K *>(shared_storage + cosize(smem_layout_A) * sizeof(ElementA));

    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    constexpr int TotalStages = StageCount::value;
    using barrier_t = cuda::barrier<cuda::thread_scope_block>;

    alignas(64) __shared__ uint8_t barrier_storage_full[sizeof(barrier_t) * TotalStages];
    alignas(64) __shared__ uint8_t barrier_storage_empty[sizeof(barrier_t) * TotalStages];

    const auto full_barriers = reinterpret_cast<barrier_t *>(barrier_storage_full);
    const auto empty_barriers = reinterpret_cast<barrier_t *>(barrier_storage_empty);

    if (tid < TotalStages) {
        init(&full_barriers[tid], 1);
        init(&empty_barriers[tid], 1);
    }
    __syncthreads();

    auto tiled_copy_A = make_tiled_copy(
        Copy_Atom<SM90_TMA_LOAD, ElementA>{},
        Layout<Shape<_1, _1> >{},
        Layout<Shape<_1, _1> >{}
    );
    auto thr_copy_A = tiled_copy_A.get_slice(tid);

    auto tiled_mma = make_tiled_mma(
        MMA_Atom<SM120_16x8x32_TN<ElementC, float_e2m1_t, float_e2m1_t> >{},
        Layout<Shape<_2, _2, _1> >{}
    );
    auto thr_mma = tiled_mma.get_slice(tid);

    auto accumulators = partition_fragment_C(tiled_mma, make_shape(TileM{}, TileN{}));
    clear(accumulators);

    const int k_tiles = size<1>(lA);

    if (warp_id == 3) {
        int write_stage = 0;

        for (int k = 0; k < k_tiles; ++k) {
            empty_barriers[write_stage].arrive_and_wait();

            if (lane_id == 0) {
                Tensor cA = thr_copy_A.get_container(lA(_, k), sA(_, _, write_stage));
                copy(tiled_copy_A, cA);
            }

            const int b_gmem_idx = block_n * blocks_per_tile_k + k * blocks_per_tile_k;
            if (tid < blocks_per_tile_k * 32) {
                const int local_block = tid / 32;
                if (lane_id < sizeof(BlockQ4K) / sizeof(uint32_t)) {
                    const auto src = reinterpret_cast<const uint32_t *>(&ptr_B_q4[b_gmem_idx + local_block]);
                    const auto dst = reinterpret_cast<uint32_t *>(&sB_q4[write_stage * blocks_per_tile_k + local_block]);
                    dst[lane_id] = src[lane_id];
                }
            }

            __syncthreads();
            if (lane_id == 0) {
                full_barriers[write_stage].arrive();
            }

            write_stage++;
            if (write_stage >= TotalStages) write_stage = 0;
        }
    } else {
        int read_stage = 0;

        for (int k = 0; k < k_tiles; ++k) {
            full_barriers[read_stage].arrive_and_wait();

            Tensor tCsA = thr_mma.partition_A(sA(_, _, read_stage));
            Tensor rA_in = thr_mma.make_fragment_A(tCsA);
            copy(tiled_mma, tCsA, rA_in);

            auto rA_fp4 = make_fragment_like<float_e2m1_t>(rA_in);
            for (int i = 0; i < size(rA_in); ++i) {
                rA_fp4(i) = float_e2m1_t(__bfloat162float(rA_in(i)));
            }

            auto rB_fp4 = thr_mma.make_fragment_B(thr_mma.partition_B(sA(_, _, read_stage)));

            for (int n_idx = 0; n_idx < TileN::value; ++n_idx) {
                for (int k_idx = 0; k_idx < TileK::value; ++k_idx) {
                    const int global_k = k * TileK::value + k_idx;
                    const int block_k_idx = global_k / 256;
                    const int local_k_idx = global_k % 256;

                    const int b_smem_idx = read_stage * blocks_per_tile_k + block_k_idx % blocks_per_tile_k;

                    float_e2m1_t unpacked_weight;
                    dequantize_q4_k_to_fp4(sB_q4[b_smem_idx], local_k_idx, &unpacked_weight);

                    int frag_b_idx = (n_idx * TileK::value + k_idx) % size(rB_fp4);
                    if (tid % 32 == 0) {
                        rB_fp4(frag_b_idx) = unpacked_weight;
                    }
                }
            }

            gemm(tiled_mma, accumulators, rA_fp4, rB_fp4, accumulators);

            if (lane_id == 0) {
                empty_barriers[read_stage].arrive();
            }

            read_stage++;
            if (read_stage >= TotalStages) read_stage = 0;
        }

        auto tCgC = thr_mma.partition_C(lC);
        copy(accumulators, tCgC);
    }
}

void launch_dequantize_fusion_mma_gguf_q4_k(
    const void *A, const void *B_q4, void *C,
    const int M, const int N, const int K,
    const int stride_A, const int stride_C,
    cudaStream_t stream) {
    using TileM = Int<128>;
    using TileN = Int<128>;
    using TileK = Int<256>;

    using ElementA = nv_bfloat16;
    using ElementC = float;
    using StageCount = Int<2>;

    constexpr auto smem_layout_A = make_layout(make_shape(TileM{}, TileK{}, StageCount{}), LayoutRight{});
    constexpr int blocks_in_tile_k = TileK::value / 256;
    size_t smem_bytes = cosize(smem_layout_A) * sizeof(ElementA) + blocks_in_tile_k * StageCount::value * sizeof(BlockQ4K);

    dim3 grid((M + TileM::value - 1) / TileM::value, (N + TileN::value - 1) / TileN::value, 1);
    dim3 block(128, 1, 1);

    cudaFuncSetAttribute(
        reinterpret_cast<const void *>(dequantize_fusion_mma_gguf_q4_k_kernel<TileM, TileN, TileK, ElementA, ElementC, StageCount>),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes
    );

    dequantize_fusion_mma_gguf_q4_k_kernel<TileM, TileN, TileK, ElementA, ElementC, StageCount>
            <<<grid, block, smem_bytes, stream>>>(
                static_cast<const ElementA *>(A), stride_A,
                static_cast<const BlockQ4K *>(B_q4),
                static_cast<ElementC *>(C), stride_C,
                M, N, K
            );
}
