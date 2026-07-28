#include <iostream>
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/bfloat16.h"
#include "cutlass/float8.h"
#include "cutlass/float_subbyte.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM103_SUPPORTED)

extern "C" void launch_blackwell_fp4_native_gemm(
    void *output_d,
    const void *input_a,
    const void *weights_b,
    const void *scales_a,
    const void *scales_b,
    int32_t m_extent,
    int32_t n_extent,
    int32_t k_extent,
    void *stream_ptr) {
    int M = m_extent;
    int N = n_extent;
    int K = k_extent;
    int batch = 1;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    if (M == 0 || N == 0 || K == 0) return;

    using ElementA = cutlass::float_e2m1_t;
    using ElementSFA = cutlass::float_ue4m3_t;
    using LayoutATag = cutlass::layout::RowMajor;
    constexpr int AlignmentA = 256;

    using ElementB = cutlass::float_e2m1_t;
    using ElementSFB = cutlass::float_ue4m3_t;
    using LayoutBTag = cutlass::layout::ColumnMajor;
    constexpr int AlignmentB = 256;

    using ElementD = cutlass::bfloat16_t;
    using ElementC = cutlass::bfloat16_t;
    using LayoutCTag = cutlass::layout::RowMajor;
    using LayoutDTag = cutlass::layout::RowMajor;
    constexpr int AlignmentD = 8;
    constexpr int AlignmentC = 8;
    using ElementAccumulator = float;

    using ArchTag = cutlass::arch::Sm103;
    using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

    using MmaTileShape1Sm = cute::Shape<cute::_128, cute::_256, Int<768> >;
    using ClusterShape = cute::Shape<int, int, cute::_1>;

    using CollectiveEpilogue1Sm = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        MmaTileShape1Sm, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutCTag, AlignmentC,
        ElementD, LayoutDTag, AlignmentD,
        cutlass::epilogue::NoSmemWarpSpecialized1Sm
    >::CollectiveOp;

    using CollectiveMainloop1Sm = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        cute::tuple<ElementA, ElementSFA>, LayoutATag, AlignmentA,
        cute::tuple<ElementB, ElementSFB>, LayoutBTag, AlignmentB,
        ElementAccumulator,
        MmaTileShape1Sm, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue1Sm::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs16Sm103
    >::CollectiveOp;

    using GemmKernel1Sm = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop1Sm,
        CollectiveEpilogue1Sm
    >;

    using Gemm1Sm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel1Sm>;
    using Sm1xxBlkScaledConfig = typename GemmKernel1Sm::CollectiveMainloop::Sm1xxBlkScaledConfig;

    auto stride_A = cutlass::make_cute_packed_stride(typename GemmKernel1Sm::StrideA{}, {M, K, batch});
    auto stride_B = cutlass::make_cute_packed_stride(typename GemmKernel1Sm::StrideB{}, {N, K, batch});
    auto stride_C = cutlass::make_cute_packed_stride(typename GemmKernel1Sm::StrideC{}, {M, N, batch});
    auto stride_D = cutlass::make_cute_packed_stride(typename GemmKernel1Sm::StrideD{}, {M, N, batch});

    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, batch));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, batch));

    float alpha = 1.0f;
    float beta = 0.0f;

    typename Gemm1Sm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, batch},
        {
            reinterpret_cast<ElementA *>(const_cast<void *>(input_a)), stride_A,
            reinterpret_cast<ElementB *>(const_cast<void *>(weights_b)), stride_B,
            reinterpret_cast<ElementSFA *>(const_cast<void *>(scales_a)), layout_SFA,
            reinterpret_cast<ElementSFB *>(const_cast<void *>(scales_b)), layout_SFB
        },
        {
            {alpha, beta},
            nullptr, stride_C,
            reinterpret_cast<ElementD *>(output_d), stride_D
        }
    };

    int cluster_m = (M >= 256) ? 2 : 1;
    int cluster_n = 1;

    args.scheduler.max_swizzle_size = 0;
    args.hw_info.cluster_shape = dim3(cluster_m, cluster_n, 1);
    args.hw_info.cluster_shape_fallback = dim3(1, 1, 1);

    Gemm1Sm gemm_op;

    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "[ENGINE ERROR] Layout/alignment mismatch or unsupported shapes." << std::endl;
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

#endif
