#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/bfloat16.h"
#include "cutlass/float8.h"
#include "cutlass/float_subbyte.h"

using namespace cute;

template<typename T>
struct KernelTraits;

template<>
struct KernelTraits<cutlass::bfloat16_t> {
    using MmaTileShape = Shape<_128, _128, _128>;
    using ClusterShape = Shape<_1, _1, _1>;
    using PerSmTileShape_MNK = Shape<_128, _128, _128>;
};

template<typename T>
struct Fp4GemmSm120 {
    using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using LayoutATag = cutlass::layout::RowMajor;
    static constexpr int AlignmentA = 32;

    using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using LayoutBTag = cutlass::layout::ColumnMajor;
    static constexpr int AlignmentB = 32;

    using ElementD = T;
    using ElementC = T;
    using LayoutCTag = cutlass::layout::RowMajor;
    using LayoutDTag = cutlass::layout::RowMajor;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    using ElementAccumulator = float;

    using ArchTag = cutlass::arch::Sm120;
    using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

    using MmaTileShape = typename KernelTraits<T>::MmaTileShape;
    using ClusterShape = typename KernelTraits<T>::ClusterShape;
    using PerSmTileShape_MNK = typename KernelTraits<T>::PerSmTileShape_MNK;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, PerSmTileShape_MNK, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutCTag, AlignmentC,
        ElementD, LayoutDTag, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB, LayoutBTag, AlignmentB,
        ElementAccumulator, MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
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
    using GemmOp = Fp4GemmSm120<cutlass::bfloat16_t>;
    using ElementA = typename GemmOp::Gemm::ElementA;
    using ElementB = typename GemmOp::Gemm::ElementB;
    using ElementSFA = cutlass::float_ue4m3_t;
    using ElementSFB = cutlass::float_ue4m3_t;
    using ElementD = typename GemmOp::Gemm::ElementD;
    using StrideA = typename GemmOp::StrideA;
    using StrideB = typename GemmOp::StrideB;
    using StrideD = typename GemmOp::StrideD;
    using Sm1xxBlkScaledConfig = typename GemmOp::Sm1xxBlkScaledConfig;

    int m = m_extent;
    int n = n_extent;
    int k = k_extent;
    auto stream = static_cast<cudaStream_t>(stream_ptr);

    if (m == 0 || n == 0 || k == 0) return;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

    typename GemmOp::Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        {
            static_cast<ElementA const *>(input_a), stride_A,
            static_cast<ElementB const *>(weights_b), stride_B,
            static_cast<ElementSFA const *>(scales_a), layout_SFA,
            static_cast<ElementSFB const *>(scales_b), layout_SFB
        },
        {
            {},
            static_cast<ElementD const *>(nullptr), stride_D,
            static_cast<ElementD *>(output_d), stride_D
        }
    };

    typename GemmOp::Gemm gemm;

    size_t workspace_size = GemmOp::Gemm::get_workspace_size(arguments);
    void *workspace = nullptr;
    if (workspace_size > 0) {
        if (cudaMalloc(&workspace, workspace_size) != cudaSuccess) return;
    }

    if (gemm.initialize(arguments, workspace) != cutlass::Status::kSuccess) {
        if (workspace) cudaFree(workspace);
        return;
    }

    gemm.run(stream);

    if (workspace) {
        cudaFree(workspace);
    }
}
