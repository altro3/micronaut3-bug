#pragma once
#include <stdint.h>
#include "core_api.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/bfloat16.h"
#include "cutlass/float8.h"

using namespace cute;

template<typename T>
struct KernelTraits;

template<>
struct KernelTraits<bfloat16_t> {
    using MmaTileShape = Shape<_128, _128, _128>;
    using ClusterShape = Shape<_1, _1, _1>;
    using PerSmTileShape_MNK = Shape<_128, _128, _128>;
};

template<typename T>
struct Fp4GemmSm120 {
    using ElementA = float_e2m1_t;
    using LayoutATag = cutlass::layout::RowMajor;
    static constexpr int AlignmentA = 32;

    using ElementB = float_e2m1_t;
    using LayoutBTag = cutlass::layout::ColumnMajor;
    static constexpr int AlignmentB = 32;

    using ElementScale = float_ue4m3_t;

    using ElementD = T;
    using ElementC = T;
    using LayoutCTag = cutlass::layout::RowMajor;
    using LayoutDTag = cutlass::layout::RowMajor;

    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    using ElementAccumulator = float;

    using ArchTag = cutlass::arch::Sm120;
    using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

    using MmaTileShape = KernelTraits<T>::MmaTileShape;
    using ClusterShape = KernelTraits<T>::ClusterShape;
    using PerSmTileShape_MNK = KernelTraits<T>::PerSmTileShape_MNK;

    using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, PerSmTileShape_MNK, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutCTag, AlignmentC,
        ElementD, LayoutDTag, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

    using CollectiveMainloop = cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        tuple<ElementA, ElementScale>, LayoutATag, AlignmentA,
        tuple<ElementB, ElementScale>, LayoutBTag, AlignmentB,
        ElementAccumulator, MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAuto,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    using StrideA = GemmKernel::StrideA;
    using StrideB = GemmKernel::StrideB;
    using StrideD = GemmKernel::StrideD;
    using Sm1xxBlkScaledConfig = CollectiveMainloop::Sm1xxBlkScaledConfig;
};

extern "C" KERNEL_API void launch_blackwell_fp4_native_gemm(
    void *output_d,
    const void *input_a,
    const void *weights_b,
    const void *scales_a,
    const void *scales_b,
    int32_t m_extent,
    int32_t n_extent,
    int32_t k_extent,
    void *stream_ptr
);
