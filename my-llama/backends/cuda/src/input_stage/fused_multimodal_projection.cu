#include <cuda_runtime.h>
#include <stdint.h>

#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/epilogue/thread/linear_combination_gelu.h"

extern "C" void launch_fused_multimodal_projection(
    const void ** __restrict__ device_ptr_A,
    const void ** __restrict__ device_ptr_B,
    void ** __restrict__ device_ptr_D,
    const float * __restrict__ bias,
    const cutlass::gemm::GemmCoord * __restrict__ device_problem_shapes,
    int32_t num_segments,
    const int32_t vision_hidden_size,
    const int32_t text_hidden_size,
    const int32_t tp_rank,
    const int32_t tp_size,
    void * __restrict__ workspace_ptr,
    cudaStream_t stream
) {
    const int32_t local_out_features = text_hidden_size / tp_size;
    const int32_t rank_offset_out_features = tp_rank * local_out_features;

    using ElementA = cutlass::bfloat16_t;
    using LayoutA = cutlass::layout::RowMajor;
    using ElementB = cutlass::bfloat16_t;
    using LayoutB = cutlass::layout::ColumnMajor;
    using ElementC = cutlass::bfloat16_t;
    using LayoutC = cutlass::layout::RowMajor;

    using ElementAccumulator = float;
    using ElementCompute = float;

    using ArchTag = cutlass::arch::Sm80;
    using OperatorClass = cutlass::arch::OpClassTensorOp;

    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

    // Эпилог слияния операций Bias + GELU
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationGELU<
        ElementC,
        128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator,
        ElementCompute
    >;

    using GemmUniversalOp = cutlass::gemm::device::GemmUniversal<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAccumulator,
        OperatorClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueOutputOp
    >;

    auto *host_problem_shapes = new cutlass::gemm::GemmCoord[num_segments];
    cudaMemcpyAsync(host_problem_shapes, device_problem_shapes, num_segments * sizeof(cutlass::gemm::GemmCoord), cudaMemcpyDeviceToHost, stream);

    auto *host_ptr_A = new const ElementA *[num_segments];
    auto *host_ptr_B = new const ElementB *[num_segments];
    auto *host_ptr_D = new ElementC *[num_segments];
    cudaMemcpyAsync(host_ptr_A, device_ptr_A, num_segments * sizeof(ElementA *), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(host_ptr_B, device_ptr_B, num_segments * sizeof(ElementB *), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(host_ptr_D, device_ptr_D, num_segments * sizeof(ElementC *), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    GemmUniversalOp gemm_op;
    auto current_workspace = static_cast<uint8_t *>(workspace_ptr);

    const EpilogueOutputOp::Params epilogue_args(
        1.0f,
        1.0f
    );

    for (int32_t i = 0; i < num_segments; ++i) {
        cutlass::gemm::GemmCoord problem_size = host_problem_shapes[i];
        if (problem_size.m() <= 0) continue;

        GemmUniversalOp::Arguments arguments(
            cutlass::gemm::GemmUniversalMode::kGemm,
            problem_size,
            1,
            epilogue_args,
            const_cast<ElementA *>(host_ptr_A[i]),
            const_cast<ElementB *>(host_ptr_B[i]),
            const_cast<float *>(bias) + rank_offset_out_features,
            host_ptr_D[i],
            problem_size.m() * problem_size.k(),
            problem_size.n() * problem_size.k(),
            0,
            problem_size.m() * problem_size.n(),
            vision_hidden_size,
            vision_hidden_size,
            0,
            local_out_features
        );

        gemm_op.initialize(arguments, current_workspace, stream);
        gemm_op.run(stream);

        size_t workspace_size = gemm_op.get_workspace_size(arguments);
        current_workspace += workspace_size;
    }

    delete[] host_problem_shapes;
    delete[] host_ptr_A;
    delete[] host_ptr_B;
    delete[] host_ptr_D;
}
