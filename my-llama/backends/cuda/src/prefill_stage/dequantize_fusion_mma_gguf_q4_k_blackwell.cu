#include "dequantize_fusion_mma_gguf_q4_k_blackwell.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
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
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;
using ElementD = bfloat16_t;
using ElementC = bfloat16_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm103;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using MmaTileShape1Sm = Shape<_128, _256, Int<768> >;
using ClusterShape = Shape<int, int, _1>;

using CollectiveEpilogue1Sm = cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape1Sm, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::NoSmemWarpSpecialized1Sm
>::CollectiveOp;

using CollectiveMainloop1Sm = cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    tuple<ElementA, ElementSFA>, LayoutATag, AlignmentA,
    tuple<ElementB, ElementSFB>, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShape1Sm, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(CollectiveEpilogue1Sm::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs16Sm103
>::CollectiveOp;

using GemmKernel1Sm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop1Sm,
    CollectiveEpilogue1Sm
>;

using Gemm1Sm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel1Sm>;

template<class LayoutSFBType>
__global__ void preprocess_gguf_q4_k_to_mx_fp4_kernel(
    const BlockQ4K *ptr_B_q4,
    ElementB *ptr_B_out,
    ElementSFB *ptr_SFB_out,
    LayoutSFBType layout_SFB,
    int N, int K) {
    int global_n = blockIdx.x * blockDim.x + threadIdx.x;
    int global_k = blockIdx.y * blockDim.y + threadIdx.y;

    if (global_n >= N || global_k >= K) return;

    int total_blocks_k = K / 256;
    int block_k_idx = global_k / 256;
    int local_k_idx = global_k % 256;
    int b_gmem_idx = global_n * total_blocks_k + block_k_idx;

    const BlockQ4K &block = ptr_B_q4[b_gmem_idx];

    int sub_block_idx = local_k_idx / 32;
    int element_idx = local_k_idx % 32;

    float d_val = __bfloat162float(block.d);
    float dmin_val = __bfloat162float(block.dmin);

    uint8_t sc_byte = block.scales[sub_block_idx * 2 + element_idx / 16];
    float scale = element_idx % 16 < 8 ? sc_byte & 0x0F : sc_byte >> 4;

    uint8_t q_byte = block.qs[(sub_block_idx * 32 + element_idx) / 2];
    uint8_t q_raw = element_idx % 2 == 0 ? q_byte & 0x0F : q_byte >> 4;

    float dequantized_f32 = d_val * scale * q_raw - dmin_val;

    int out_weight_idx = global_n * K + global_k;
    ptr_B_out[out_weight_idx] = ElementB(dequantized_f32);

    if (global_k % 32 == 0) {
        int logical_k_sf = global_k / 32;
        auto sf_coord = make_coord(global_n, logical_k_sf, 0);
        int packed_sf_idx = layout_SFB(sf_coord);
        ptr_SFB_out[packed_sf_idx] = ElementSFB(scale * d_val);
    }
}

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

    std::cout << "[DIAGNOSTIC] Problem shapes submitted -> M: " << M << ", N: " << N << ", K: " << K << std::endl;

    if (M == 0 || N == 0 || K == 0) {
        std::cerr << "[DIAGNOSTIC ERROR] Matrix dimension is zero! Core dump prevented." << std::endl;
        return;
    }

    using Sm1xxBlkScaledConfig = GemmKernel1Sm::CollectiveMainloop::Sm1xxBlkScaledConfig;

    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

    std::cout << "[DIAGNOSTIC] layout_SFA total logical size: " << size(layout_SFA) << std::endl;
    std::cout << "[DIAGNOSTIC] layout_SFB total logical size: " << size(layout_SFB) << std::endl;
    std::cout << "[DIAGNOSTIC] layout_SFA physical size (filtered): " << size(filter_zeros(layout_SFA)) << std::endl;
    std::cout << "[DIAGNOSTIC] layout_SFB physical size (filtered): " << size(filter_zeros(layout_SFB)) << std::endl;

    if (size(filter_zeros(layout_SFA)) == 0 || size(filter_zeros(layout_SFB)) == 0) {
        std::cerr << "[DIAGNOSTIC ERROR] Block-scaled layout generation returned size 0. Check tile dimensions and K constraints!" << std::endl;
        return;
    }

    static ElementB *d_processed_B = nullptr;
    static ElementSFB *d_processed_SFB = nullptr;
    static ElementSFA *d_processed_SFA = nullptr;
    static int current_M = 0;
    static int current_N = 0;
    static int current_K = 0;

    if (d_processed_B == nullptr || current_M != M || current_N != N || current_K != K) {
        if (d_processed_B) {
            cudaFree(d_processed_B);
            cudaFree(d_processed_SFB);
            cudaFree(d_processed_SFA);
        }
        cudaMalloc(&d_processed_B, N * K * sizeof(ElementB));
        cudaMalloc(&d_processed_SFB, size(filter_zeros(layout_SFB)) * sizeof(ElementSFB));
        cudaMalloc(&d_processed_SFA, size(filter_zeros(layout_SFA)) * sizeof(ElementSFA));

        cudaMemset(d_processed_SFA, 0x3C, size(filter_zeros(layout_SFA)) * sizeof(ElementSFA));

        current_M = M;
        current_N = N;
        current_K = K;
    }

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (K + 15) / 16);
    preprocess_gguf_q4_k_to_mx_fp4_kernel<<<grid, block, 0, stream>>>(
        static_cast<const BlockQ4K *>(quantized_weights),
        d_processed_B,
        d_processed_SFB,
        layout_SFB,
        N, K
    );

    auto stride_A = cutlass::make_cute_packed_stride(GemmKernel1Sm::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(GemmKernel1Sm::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(GemmKernel1Sm::StrideC{}, {M, N, 1});

    GemmKernel1Sm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            static_cast<ElementA const *>(input_activations), stride_A,
            static_cast<ElementB const *>(d_processed_B), stride_B,
            static_cast<ElementSFA const *>(d_processed_SFA), layout_SFA,
            static_cast<ElementSFB const *>(d_processed_SFB), layout_SFB
        },
        {
            {1.0f, 0.0f},
            static_cast<ElementC *>(output_activations), stride_C,
            static_cast<ElementD *>(output_activations), stride_C
        }
    };

    Gemm1Sm gemm_op;

    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "[DIAGNOSTIC ERROR] gemm_op.can_implement failed! CUTLASS Status Code: "
                << cutlass::cutlassGetStatusString(status) << std::endl;
        return;
    }

    size_t workspace_size = Gemm1Sm::get_workspace_size(args);
    void *workspace_ptr = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace_ptr, workspace_size);
    }

    status = gemm_op.run(args, workspace_ptr, stream);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "[DIAGNOSTIC ERROR] gemm_op.run critical execution failure! Code: "
                << cutlass::cutlassGetStatusString(status) << std::endl;
    }

    if (workspace_ptr) {
        cudaFree(workspace_ptr);
    }
}

#endif
