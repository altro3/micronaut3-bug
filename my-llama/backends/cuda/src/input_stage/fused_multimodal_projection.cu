#include <cuda_runtime.h>
#include <stdint.h>

#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/epilogue/thread/linear_combination_gelu.h"

extern "C" void launch_fused_multimodal_projection(
    void * __restrict__ out_tokens,
    const void * __restrict__ input_tokens,
    const void * __restrict__ weight_matrix,
    const float * __restrict__ bias,
    const int32_t * __restrict__ vision_segments,
    int32_t num_segments,
    const int32_t vision_hidden_size,
    const int32_t text_hidden_size,
    const int32_t tp_rank,
    const int32_t tp_size,
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

    const auto host_segments = new int32_t[num_segments * 2];
    cudaMemcpyAsync(host_segments, vision_segments, num_segments * 2 * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    GemmUniversalOp gemm_op;

    for (int32_t i = 0; i < num_segments; ++i) {
        const int32_t start_token = host_segments[i * 2];
        const int32_t end_token = host_segments[i * 2 + 1];
        int32_t segment_tokens = end_token - start_token;

        if (segment_tokens <= 0) continue;

        auto *ptr_A = const_cast<ElementA *>(static_cast<const ElementA *>(input_tokens)) + start_token * vision_hidden_size;
        auto *ptr_B = const_cast<ElementB *>(static_cast<const ElementB *>(weight_matrix)) + rank_offset_out_features * vision_hidden_size;
        auto *ptr_D = static_cast<ElementC *>(out_tokens) + start_token * local_out_features;

        cutlass::gemm::GemmCoord problem_size(segment_tokens, local_out_features, vision_hidden_size);

        EpilogueOutputOp::Params epilogue_args(
            1.0f,
            1.0f
        );

        GemmUniversalOp::Arguments arguments(
            cutlass::gemm::GemmUniversalMode::kGemm,
            problem_size,
            1,
            epilogue_args,
            ptr_A,
            ptr_B,
            const_cast<float *>(bias) + rank_offset_out_features,
            ptr_D,
            problem_size.m() * problem_size.k(),
            problem_size.n() * problem_size.k(),
            0,
            problem_size.m() * problem_size.n(),
            vision_hidden_size,
            vision_hidden_size,
            0,
            local_out_features
        );

        size_t workspace_size = gemm_op.get_workspace_size(arguments);
        void *workspace = nullptr;
        if (workspace_size > 0) {
            cudaMallocAsync(&workspace, workspace_size, stream);
        }

        gemm_op.initialize(arguments, workspace, stream);
        gemm_op.run(stream);

        if (workspace) {
            cudaFreeAsync(workspace, stream);
        }
    }

    delete[] host_segments;
}
