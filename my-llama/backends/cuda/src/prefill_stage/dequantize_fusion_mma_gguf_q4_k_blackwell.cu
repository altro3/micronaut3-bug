#include "dequantize_fusion_mma_gguf_q4_k_blackwell.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/util/packed_stride.hpp"

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
    using ElementA = typename GemmOp::ElementA;
    using ElementB = typename GemmOp::ElementB;
    using ElementSFA = typename GemmOp::ElementScale;
    using ElementSFB = typename GemmOp::ElementScale;
    using ElementD = typename GemmOp::ElementD;
    using StrideA = typename GemmOp::StrideA;
    using StrideB = typename GemmOp::StrideB;
    using StrideD = typename GemmOp::StrideD;
    using Sm1xxBlkScaledConfig = typename GemmOp::Sm1xxBlkScaledConfig;

    int m = m_extent;
    int n = n_extent;
    int k = k_extent;
    auto stream = static_cast<cudaStream_t>(stream_ptr);

    if (m == 0 || n == 0 || k == 0) return;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(m, k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(n, k, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, make_shape(m, n, 1));

    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));

#if defined(_MSC_VER)
#define ALIGN_32 __declspec(align(32))
#else
#define ALIGN_32 alignas(32)
#endif

    ALIGN_32 typename GemmOp::Gemm::Arguments arguments{
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

    int current_device = 0;
    cudaGetDevice(&current_device);
    int actual_sm_count = 0;
    cudaDeviceGetAttribute(&actual_sm_count, cudaDevAttrMultiProcessorCount, current_device);

    arguments.hw_info.sm_count = actual_sm_count;
    arguments.hw_info.max_active_clusters = 1;
    arguments.hw_info.cluster_shape = dim3(1, 1, 1);

    GemmOp::Gemm gemm;

    cutlass::Status status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "[ENGINE ERROR] can_implement failed: " << cutlass::cutlassGetStatusString(status)
                << " (Detected SMs: " << actual_sm_count << ")" << std::endl;
        return;
    }

    size_t workspace_size = GemmOp::Gemm::get_workspace_size(arguments);
    void *workspace = nullptr;
    if (workspace_size > 0) {
        if (cudaMalloc(&workspace, workspace_size) != cudaSuccess) return;
    }

    status = gemm.initialize(arguments, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "[ENGINE ERROR] Initialize failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
        if (workspace) cudaFree(workspace);
        return;
    }

    status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "[ENGINE ERROR] Run failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
    }

    if (workspace) {
        cudaFree(workspace);
    }
}
