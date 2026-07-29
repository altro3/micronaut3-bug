#include <iostream>
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/bfloat16.h"
#include "cutlass/float8.h"
#include "cutlass/float_subbyte.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include <cutlass/util/packed_stride.hpp>

using namespace cute;

#ifndef CUTLASS_ARCH_MMA_SM120_SUPPORTED
#define CUTLASS_ARCH_MMA_SM120_SUPPORTED 1
#endif

template<typename T>
struct KernelTraits;

template<>
struct KernelTraits<bfloat16_t> {
    using MmaTileShape = Shape<_128, _128, _128>;
    using ClusterShape = Shape<_1, _1, _1>;
    using PerSmTileShape_MNK = Shape<_128, _128, _128>;
};

extern "C" void launch_blackwell_fp4_native_gemm(
    void *output_d,
    const void *input_a,
    const void *weights_b,
    const void *scales_a,
    const void *scales_b,
    int32_t m_extent,
    int32_t n_extent,
    int32_t k_extent,
    void *stream_ptr
) {
    int M = m_extent;
    int N = n_extent;
    int K = k_extent;
    int batch = 1;
    auto stream = static_cast<cudaStream_t>(stream_ptr);

    if (M == 0 || N == 0 || K == 0) return;

    using ElementA = cutlass::nv_float4_t<float_e2m1_t>;
    using LayoutATag = cutlass::layout::RowMajor;
    constexpr int AlignmentA = 32;

    using ElementB = cutlass::nv_float4_t<float_e2m1_t>;
    using LayoutBTag = cutlass::layout::ColumnMajor;
    constexpr int AlignmentB = 32;

    using ElementD = bfloat16_t;
    using ElementC = bfloat16_t;
    using LayoutCTag = cutlass::layout::RowMajor;
    using LayoutDTag = cutlass::layout::RowMajor;
    constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
    constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    using ElementAccumulator = float;

    using ArchTag = cutlass::arch::Sm120;
    using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

    using Traits = KernelTraits<bfloat16_t>;
    using MmaTileShape = Traits::MmaTileShape;
    using ClusterShape = Traits::ClusterShape;
    using PerSmTileShape_MNK = Traits::PerSmTileShape_MNK;

    using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        PerSmTileShape_MNK, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutCTag, AlignmentC,
        ElementD, LayoutDTag, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

    using CollectiveMainloop = cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, LayoutATag, AlignmentA,
        ElementB, LayoutBTag, AlignmentB,
        ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue
    >;

    using Gemm1Sm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using Sm1xxBlkScaledConfig = GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

    using StrideA = GemmKernel::StrideA;
    using StrideB = GemmKernel::StrideB;
    using StrideC = GemmKernel::StrideC;
    using StrideD = GemmKernel::StrideD;

    using ElementSFA = float_ue4m3_t;
    using ElementSFB = float_ue4m3_t;

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, batch});
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, batch});
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, batch});
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, batch});

    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, batch));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, batch));

    float alpha = 1.0f;
    float beta = 0.0f;

    using InternalElementA = GemmKernel::CollectiveMainloop::ElementA;
    using InternalElementB = GemmKernel::CollectiveMainloop::ElementB;

    Gemm1Sm::Arguments args{
        .mode = cutlass::gemm::GemmUniversalMode::kGemm,
        .problem_shape = {M, N, K, batch},
        .mainloop = {
            .ptr_A = static_cast<InternalElementA const *>(input_a), .dA = stride_A,
            .ptr_B = static_cast<InternalElementB const *>(weights_b), .dB = stride_B,
            .ptr_SFA = static_cast<ElementSFA const *>(scales_a), .layout_SFA = layout_SFA,
            .ptr_SFB = static_cast<ElementSFB const *>(scales_b), .layout_SFB = layout_SFB
        },
        .epilogue = {
            .thread = {.alpha = alpha, .beta = beta},
            .ptr_C = nullptr, .dC = stride_C,
            .ptr_D = static_cast<ElementD *>(output_d), .dD = stride_D
        }
    };


    args.scheduler.max_swizzle_size = 1;
    args.hw_info.cluster_shape = dim3(1, 1, 1);

    Gemm1Sm gemm_op;

    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "[ENGINE ERROR] can_implement failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
        return;
    }

    size_t workspace_size = gemm_op.get_workspace_size(args);
    uint8_t *workspace_ptr = nullptr;
    if (workspace_size > 0) {
        cudaMallocAsync(&workspace_ptr, workspace_size, stream);
    }

    status = gemm_op.initialize(args, workspace_ptr, stream);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "[ENGINE ERROR] Initialize failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
        if (workspace_ptr) cudaFreeAsync(workspace_ptr, stream);
        return;
    }

    status = gemm_op.run(stream);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "[ENGINE ERROR] Run failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
    }

    if (workspace_ptr) {
        cudaFreeAsync(workspace_ptr, stream);
    }
}
