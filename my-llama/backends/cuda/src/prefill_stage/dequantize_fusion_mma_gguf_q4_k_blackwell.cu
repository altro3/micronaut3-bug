#include "dequantize_fusion_mma_gguf_q4_k_blackwell.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM103_SUPPORTED)

using ElementA = float_e2m1_t;
using ElementSFA = float_ue4m3_t;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

using ElementB = float_e2m1_t;
using ElementSFB = float_ue4m3_t;
using LayoutBTag = cutlass::layout::RowMajor;
constexpr int AlignmentB = 32;

using ElementD = bfloat16_t;
using ElementC = bfloat16_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentD = 8;
constexpr int AlignmentC = 8;
using ElementAccumulator = float;

using ArchTag = cutlass::arch::Sm103;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using MmaTileShape1Sm = Shape<_128, _256, Int<768> >;
using ClusterShape = Shape<int, int, _1>;

template<class TileShape>
struct GgufQ4KToSmemPipeline {
    struct SharedStorage {
        alignas(128) ElementA smem_A[size<0>(TileShape{}) * size<2>(TileShape{})];
        alignas(128) ElementB smem_B[size<1>(TileShape{}) * size<2>(TileShape{})];
        alignas(128) ElementSFA smem_SFA[size<0>(TileShape{}) * (size<2>(TileShape{}) / 32)];
        alignas(128) ElementSFB smem_SFB[size<1>(TileShape{}) * (size<2>(TileShape{}) / 32)];
    };

    struct Params {
        BlockQ4K const *ptr_B_q4;
        int K;
        int N;
        int total_blocks_k;
    };

    __device__ static void load_and_dequantize(
        Params const &params,
        SharedStorage &shared_storage,
        int thread_idx,
        int block_n,
        int block_k) {
        int lane_id = thread_idx % 32;
        int warp_id = thread_idx / 32;

        if (warp_id < 4) {
            int tid = warp_id * 32 + lane_id;
            int total_threads = 4 * 32;
            int tile_n = size<1>(TileShape{});
            int tile_k = size<2>(TileShape{});
            int elements_per_tile = tile_n * tile_k;

            for (int idx = tid; idx < elements_per_tile; idx += total_threads) {
                int local_n = idx / tile_k;
                int local_k = idx % tile_k;

                int global_n = block_n + local_n;
                int global_k = block_k + local_k;

                if (global_n < params.N && global_k < params.K) {
                    int block_k_idx = global_k / 256;
                    int local_k_idx = global_k % 256;

                    int b_gmem_idx = global_n * params.total_blocks_k + block_k_idx;

                    const BlockQ4K &block = params.ptr_B_q4[b_gmem_idx];

                    int sub_block_idx = local_k_idx / 32;
                    int element_idx = local_k_idx % 32;

                    float d_val = __bfloat162float(block.d);
                    float dmin_val = __bfloat162float(block.dmin);

                    uint8_t sc_byte = block.scales[sub_block_idx * 2 + element_idx / 16];
                    float scale = element_idx % 16 < 8 ? sc_byte & 0x0F : sc_byte >> 4;

                    uint8_t q_byte = block.qs[(sub_block_idx * 32 + element_idx) / 2];
                    uint8_t q_raw = element_idx % 2 == 0 ? q_byte & 0x0F : q_byte >> 4;

                    float dequantized_f32 = d_val * scale * q_raw - dmin_val;

                    if (blockIdx.x == 0 && blockIdx.y == 0 && local_n == 0 && local_k == 0) {
                        printf("[GPU 4.6.1 MATH] b_gmem_idx: %d, d_val: %f, scale: %f, q_raw: %d, unpacked: %f\n",
                               b_gmem_idx, d_val, scale, q_raw, dequantized_f32);
                    }

                    shared_storage.smem_B[local_n * tile_k + local_k] = ElementB(dequantized_f32);

                    if (local_k % 32 == 0) {
                        int local_k_sf = local_k / 32;
                        shared_storage.smem_SFB[local_n * (tile_k / 32) + local_k_sf] = ElementSFB(scale * d_val);
                    }
                }
            }
        }
    }
};

using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape1Sm, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::NoSmemWarpSpecialized1Sm
>::CollectiveOp;

using CollectiveMainloop = cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    tuple<ElementA, ElementSFA>, LayoutATag, AlignmentA,
    tuple<ElementB, ElementSFB>, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShape1Sm, ClusterShape,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::KernelTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs16Sm103
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using GemmAdapter = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

extern "C" void launch_fused_gemm_gguf_blackwell_fp4_native(
    void *output_activations,
    const void *input_activations,
    const void *quantized_weights,
    int32_t batch_size_or_tokens,
    int32_t hidden_units_out,
    int32_t hidden_units_in,
    void *stream_ptr) {
    int M = batch_size_or_tokens;
    int N = hidden_units_out;
    int K = hidden_units_in;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    if (M == 0 || N == 0 || K == 0) return;

    std::cout << "[HOST CUTLASS 4.6.1] Launching Native GGUF Q4_K Submissions..." << std::endl;

    using BlkScaledConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;
    auto layout_SFA = BlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    auto layout_SFB = BlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

    static ElementSFB *d_processed_SFB = nullptr;
    static ElementSFA *d_processed_SFA = nullptr;
    static int current_M = 0;
    static int current_N = 0;
    static int current_K = 0;

    if (d_processed_SFA == nullptr || current_M != M || current_N != N || current_K != K) {
        if (d_processed_SFA) {
            cudaFree(d_processed_SFB);
            cudaFree(d_processed_SFA);
        }
        cudaMalloc(&d_processed_SFB, size(filter_zeros(layout_SFB)) * sizeof(ElementSFB));
        cudaMalloc(&d_processed_SFA, size(filter_zeros(layout_SFA)) * sizeof(ElementSFA));

        cudaMemset(d_processed_SFA, 0x3C, size(filter_zeros(layout_SFA)) * sizeof(ElementSFA));

        current_M = M;
        current_N = N;
        current_K = K;
    }

    auto stride_A = cutlass::make_cute_packed_stride(typename GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(typename GemmKernel::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(typename GemmKernel::StrideC{}, {M, N, 1});

    typename GemmKernel::MainloopArguments mainloop_args{
        static_cast<ElementA const *>(input_activations), stride_A,
        reinterpret_cast<ElementB const *>(quantized_weights), stride_B,
        static_cast<ElementSFA const *>(d_processed_SFA), layout_SFA,
        static_cast<ElementSFB const *>(d_processed_SFB), layout_SFB
    };

    typename GemmKernel::EpilogueArguments epilogue_args{
        {1.0f, 0.0f},
        static_cast<ElementC *>(output_activations), stride_C,
        static_cast<ElementD *>(output_activations), stride_C
    };

    typename GemmKernel::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        mainloop_args,
        epilogue_args
    };

    GemmAdapter gemm_op;

    size_t workspace_size = gemm_op.get_workspace_size(args);
    void *workspace_ptr = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace_ptr, workspace_size);
    }

    cutlass::Status status = gemm_op.run(args, workspace_ptr, stream);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "[DIAGNOSTIC ERROR] CUTLASS 4.6.1 Exec failed: "
                << cutlass::cutlassGetStatusString(status) << std::endl;
    }

    cudaStreamSynchronize(stream);
    if (workspace_ptr) {
        cudaFree(workspace_ptr);
    }
}

#endif
