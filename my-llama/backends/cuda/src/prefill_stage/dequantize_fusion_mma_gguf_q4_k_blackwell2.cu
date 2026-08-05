#define CUTLASS_DEBUG_TRACE_LEVEL 100

#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

#define CUDA_CHECK(call)                                                  \
do {                                                                    \
    cudaError_t err = call;                                               \
    if (err != cudaSuccess) {                                             \
        std::cerr << "CUDA Error in " << #call << " at " << __FILE__ << ":"  \
                  << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
        throw std::runtime_error(cudaGetErrorString(err));                  \
    }                                                                     \
} while (0)

#define CUTLASS_CHECK(status)                                                   \
do {                                                                          \
    cutlass::Status error = status;                                             \
    if (error != cutlass::Status::kSuccess) {                                   \
        std::cerr << "CUTLASS Error: " << cutlassGetStatusString(error) << " at " \
                  << __FILE__ << ":" << __LINE__ << std::endl;                    \
        throw std::runtime_error(cutlassGetStatusString(error));                  \
    }                                                                           \
} while (0)

using namespace cute;

template<typename T>
struct KernelTraits;

template<>
struct KernelTraits<float> {
    using MmaTileShape = Shape<_128, _64, _128>;
    using ClusterShape = Shape<_1, _1, _1>;
    using PerSmTileShape_MNK = Shape<_128, _64, _128>;
};

template<>
struct KernelTraits<half_t> {
    using MmaTileShape = Shape<_128, _64, _128>;
    using ClusterShape = Shape<_1, _1, _1>;
    using PerSmTileShape_MNK = Shape<_128, _64, _128>;
};

template<typename T>
struct Fp4GemmSm100 {
    using ElementA = cutlass::nv_float4_t<float_e2m1_t>;
    using LayoutATag = cutlass::layout::RowMajor;
    static constexpr int AlignmentA = 32;

    using ElementB = cutlass::nv_float4_t<float_e2m1_t>;
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

    using MmaTileShape = KernelTraits<T>::MmaTileShape;
    using ClusterShape = KernelTraits<T>::ClusterShape;
    using PerSmTileShape_MNK = KernelTraits<T>::PerSmTileShape_MNK;

    using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, PerSmTileShape_MNK, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator, ElementC, LayoutCTag, AlignmentC,
        ElementD, LayoutDTag, AlignmentD, cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

    using CollectiveMainloop = cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA,
        ElementB, LayoutBTag, AlignmentB, ElementAccumulator, MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    using StrideA = Gemm::GemmKernel::StrideA;
    using StrideB = Gemm::GemmKernel::StrideB;
    using StrideC = Gemm::GemmKernel::StrideC;
    using StrideD = Gemm::GemmKernel::StrideD;

    using Sm1xxBlkScaledConfig = CollectiveMainloop::Sm1xxBlkScaledConfig;

    static_assert(GemmKernel::SharedStorageSize <= cutlass::arch::sm100_smem_capacity_bytes,
                  "SMEM usage exceeded SM100 capacity.");
};

template<typename T>
void runGemm(int M, int N, int K,
             void *d_A, void *d_B, void *d_C, void *d_D,
             void *d_A_scale, void *d_B_scale,
             float alpha) {
    using GemmOp = Fp4GemmSm100<T>;

    using ElementA = GemmOp::Gemm::ElementA;
    using ElementB = GemmOp::Gemm::ElementB;
    using ElementSFA = cutlass::float_ue4m3_t;
    using ElementSFB = cutlass::float_ue4m3_t;
    using ElementD = GemmOp::Gemm::ElementD;
    using StrideA = GemmOp::StrideA;
    using StrideB = GemmOp::StrideB;
    using StrideD = GemmOp::StrideD;
    using Sm1xxBlkScaledConfig = GemmOp::Sm1xxBlkScaledConfig;

    int m = M;
    int n = N;
    int k = K;
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));

    using GemmArguments = GemmOp::Gemm::Arguments;
    GemmArguments *arguments_ptr = new GemmArguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            static_cast<ElementA const *>(d_A), stride_A,
            static_cast<ElementB const *>(d_B), stride_B,
            static_cast<ElementSFA const *>(d_A_scale), layout_SFA,
            static_cast<ElementSFB const *>(d_B_scale), layout_SFB
        },
        {
            {alpha, 0.0f},
            static_cast<ElementD const *>(d_C), stride_D,
            static_cast<ElementD *>(d_D), stride_D
        }
    };

    typename GemmOp::Gemm gemm;

    size_t workspace_size = GemmOp::Gemm::get_workspace_size(*arguments_ptr);
    void *workspace = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    }

    std::cout << "Initializing CUTLASS GEMM (FP4 Block-Scaled)..." << std::endl;
    CUTLASS_CHECK(gemm.initialize(*arguments_ptr, workspace));

    std::cout << "Running CUTLASS GEMM via Blackwell Async tcgen05 Engine..." << std::endl;
    CUTLASS_CHECK(gemm.run());
    std::cout << "CUTLASS GEMM Finished." << std::endl;

    if (workspace) {
        cudaFree(workspace);
    }

    delete arguments_ptr;
}

int main() {
    try {
        int M = 1024;
        int N = 4096;
        int K = 4096;
        float alpha = 1.0f;
        using OutputType = float;

        size_t size_A_bytes = M * K / 2;
        size_t size_B_bytes = K * N / 2;
        size_t size_C_bytes = M * N * sizeof(OutputType);
        size_t size_D_bytes = M * N * sizeof(OutputType);

        size_t size_A_scale_bytes = M * K / 16 * sizeof(float);
        size_t size_B_scale_bytes = K * N / 16 * sizeof(float);

        void *d_A, *d_B, *d_C, *d_D, *d_A_scale, *d_B_scale;

        std::cout << "Allocating device memory for Blackwell Native FP4 Execution..." << std::endl;
        CUDA_CHECK(cudaMalloc(&d_A, size_A_bytes));
        CUDA_CHECK(cudaMalloc(&d_B, size_B_bytes));
        CUDA_CHECK(cudaMalloc(&d_C, size_C_bytes));
        CUDA_CHECK(cudaMalloc(&d_D, size_D_bytes));
        CUDA_CHECK(cudaMalloc(&d_A_scale, size_A_scale_bytes));
        CUDA_CHECK(cudaMalloc(&d_B_scale, size_B_scale_bytes));

        CUDA_CHECK(cudaMemset(d_A, 0, size_A_bytes));
        CUDA_CHECK(cudaMemset(d_B, 0, size_B_bytes));
        CUDA_CHECK(cudaMemset(d_C, 0, size_C_bytes));
        CUDA_CHECK(cudaMemset(d_D, 0, size_D_bytes));
        CUDA_CHECK(cudaMemset(d_A_scale, 0, size_A_scale_bytes));
        CUDA_CHECK(cudaMemset(d_B_scale, 0, size_B_scale_bytes));

        runGemm<OutputType>(M, N, K, d_A, d_B, d_C, d_D, d_A_scale, d_B_scale, alpha);

        CUDA_CHECK(cudaDeviceSynchronize());
        std::cout << "Device synchronized successfully." << std::endl;

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        CUDA_CHECK(cudaFree(d_D));
        CUDA_CHECK(cudaFree(d_A_scale));
        CUDA_CHECK(cudaFree(d_B_scale));

        std::cout << "\n[SUCCESS] Standalone test finished without errors." << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "\n[FAILURE] An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
