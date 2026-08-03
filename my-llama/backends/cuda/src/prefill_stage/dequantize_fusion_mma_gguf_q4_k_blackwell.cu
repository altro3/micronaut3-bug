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
    using GemmOp = Fp4GemmSm120<bfloat16_t>;
    using GemmDeviceAdapter = GemmOp::Gemm;
    using Sm1xxBlkScaledConfig = GemmOp::Sm1xxBlkScaledConfig;

    GemmDeviceAdapter gemm;

    int m = m_extent;
    int n = n_extent;
    int k = k_extent;
    auto stream = static_cast<cudaStream_t>(stream_ptr);

    auto stride_A = cutlass::make_cute_packed_stride(GemmOp::StrideA{}, make_shape(m, k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(GemmOp::StrideB{}, make_shape(n, k, 1));
    auto stride_D = cutlass::make_cute_packed_stride(GemmOp::StrideD{}, make_shape(m, n, 1));

    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));

    GemmOp::Gemm::Arguments arguments{
        .mode = cutlass::gemm::GemmUniversalMode::kGemm,
        .problem_shape = {m, n, k, 1},
        .mainloop = {
            .ptr_A = static_cast<GemmOp::ElementA const *>(input_a),
            .dA = stride_A,
            .ptr_B = static_cast<GemmOp::ElementB const *>(weights_b),
            .dB = stride_B,
            .ptr_SFA = static_cast<GemmOp::ElementScale const *>(scales_a),
            .layout_SFA = layout_SFA,
            .ptr_SFB = static_cast<GemmOp::ElementScale const *>(scales_b),
            .layout_SFB = layout_SFB
        },
        .epilogue = {
            .thread = {},
            .ptr_C = static_cast<bfloat16_t const *>(nullptr),
            .dC = stride_D,
            .ptr_D = static_cast<bfloat16_t *>(output_d),
            .dD = stride_D
        }
    };

    int current_device = 0;
    cudaGetDevice(&current_device);
    int actual_sm_count = 0;
    cudaDeviceGetAttribute(&actual_sm_count, cudaDevAttrMultiProcessorCount, current_device);

    arguments.hw_info.sm_count = actual_sm_count;
    arguments.hw_info.max_active_clusters = 1;
    arguments.hw_info.cluster_shape = dim3(1, 1, 1);

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
