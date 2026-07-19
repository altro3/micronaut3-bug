#include <cuda_runtime.h>
#include <stdint.h>

#include "cute/tensor.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

using namespace cute;

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

    using ElementA = bfloat16_t;
    using LayoutA = cutlass::layout::RowMajor;
    using ElementB = bfloat16_t;
    using LayoutB = cutlass::layout::ColumnMajor;
    using ElementC = bfloat16_t;
    using LayoutC = cutlass::layout::RowMajor;

    using ElementAccumulator = float;
    using ElementCompute = float;

    using ArchTag = cutlass::arch::Sm120;
    using OperatorClass = cutlass::arch::OpClassTensorOp;

    using TileShape = Shape<_128, _128, _64>;
    using ClusterShape = Shape<_1, _1, _1>;

    using EVT_BiasGELU = cutlass::epilogue::fusion::Sm90TreeVisitor<
        cutlass::epilogue::fusion::Sm90Compute<
            cutlass::epilogue::thread::GELU, ElementC, ElementCompute,
            cutlass::FloatRoundStyle::round_to_nearest
        >,
        cutlass::epilogue::fusion::Sm90TreeVisitor<
            cutlass::epilogue::fusion::Sm90Compute<
                cutlass::plus, ElementCompute, ElementCompute,
                cutlass::FloatRoundStyle::round_to_nearest
            >,
            cutlass::epilogue::fusion::Sm90AccFetch,
            cutlass::epilogue::fusion::Sm90ColBroadcast<0, TileShape, ElementCompute>
        >
    >;

    using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag,
        OperatorClass,
        TileShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        ElementCompute,
        ElementC,
        LayoutC,
        8,
        ElementC,
        LayoutC,
        8,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        EVT_BiasGELU
    >::CollectiveOp;

    using CollectiveMainloop = cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120,
        cutlass::arch::OpClassTensorOp,
        ElementA, cutlass::gemm::TagToStrideA_t<LayoutA>, 8,
        ElementB, cutlass::gemm::TagToStrideB_t<LayoutB>, 8,
        ElementAccumulator,
        TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAuto,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue
    >;

    using GemmGroupedUniversal = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    const auto host_segments = new int32_t[num_segments * 2];
    cudaMemcpyAsync(host_segments, vision_segments, num_segments * 2 * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    using ProblemShapeType = Shape<int, int, int>;
    const auto host_problem_shapes = new ProblemShapeType[num_segments];

    const auto host_ptr_A = new ElementA *[num_segments];
    const auto host_ptr_B = new ElementB *[num_segments];
    const auto host_ptr_C = new ElementC *[num_segments];
    const auto host_ptr_D = new ElementC *[num_segments];

    for (int32_t i = 0; i < num_segments; ++i) {
        const int32_t start_token = host_segments[i * 2];
        const int32_t end_token = host_segments[i * 2 + 1];
        int32_t segment_tokens = end_token - start_token;

        host_problem_shapes[i] = make_shape(segment_tokens, local_out_features, vision_hidden_size);

        host_ptr_A[i] = const_cast<ElementA *>(static_cast<const ElementA *>(input_tokens)) + start_token * vision_hidden_size;
        host_ptr_B[i] = const_cast<ElementB *>(static_cast<const ElementB *>(weight_matrix)) + rank_offset_out_features * vision_hidden_size;
        host_ptr_C[i] = static_cast<ElementC *>(out_tokens) + start_token * local_out_features;
        host_ptr_D[i] = static_cast<ElementC *>(out_tokens) + start_token * local_out_features;
    }

    ProblemShapeType *device_problem_shapes;
    ElementA **device_ptr_A;
    ElementB **device_ptr_B;
    ElementC **device_ptr_C;
    ElementC **device_ptr_D;

    cudaMallocAsync(&device_problem_shapes, num_segments * sizeof(ProblemShapeType), stream);
    cudaMallocAsync(&device_ptr_A, num_segments * sizeof(ElementA *), stream);
    cudaMallocAsync(&device_ptr_B, num_segments * sizeof(ElementB *), stream);
    cudaMallocAsync(&device_ptr_C, num_segments * sizeof(ElementC *), stream);
    cudaMallocAsync(&device_ptr_D, num_segments * sizeof(ElementC *), stream);

    cudaMemcpyAsync(device_problem_shapes, host_problem_shapes, num_segments * sizeof(ProblemShapeType), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_ptr_A, host_ptr_A, num_segments * sizeof(ElementA *), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_ptr_B, host_ptr_B, num_segments * sizeof(ElementB *), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_ptr_C, host_ptr_C, num_segments * sizeof(ElementC *), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_ptr_D, host_ptr_D, num_segments * sizeof(ElementC *), cudaMemcpyHostToDevice, stream);

    GemmGroupedUniversal::Arguments arguments;

    arguments.problem_shape = make_tuple(0, 0, 0, num_segments);

    arguments.mainloop.ptr_A = reinterpret_cast<CollectiveMainloop::ElementA const *>(device_ptr_A);
    arguments.mainloop.dA = CollectiveMainloop::StrideA{};
    cute::get<0>(arguments.mainloop.dA) = static_cast<long long>(vision_hidden_size);

    arguments.mainloop.ptr_B = reinterpret_cast<CollectiveMainloop::ElementB const *>(device_ptr_B);
    arguments.mainloop.dB = CollectiveMainloop::StrideB{};
    cute::get<0>(arguments.mainloop.dB) = static_cast<long long>(vision_hidden_size);

    arguments.epilogue.thread.op_0.op_0 = {};
    arguments.epilogue.thread.op_0.op_1.ptr_col = bias + rank_offset_out_features;
    arguments.epilogue.thread.op_1 = {};

    arguments.epilogue.ptr_C = reinterpret_cast<CollectiveEpilogue::ElementC const *>(device_ptr_C);
    arguments.epilogue.dC = CollectiveEpilogue::StrideC{};
    cute::get<0>(arguments.epilogue.dC) = static_cast<long long>(local_out_features);

    arguments.epilogue.ptr_D = reinterpret_cast<CollectiveEpilogue::ElementD *>(device_ptr_D);
    arguments.epilogue.dD = CollectiveEpilogue::StrideD{};
    cute::get<0>(arguments.epilogue.dD) = static_cast<long long>(local_out_features);

    GemmGroupedUniversal gemm_op;

    size_t workspace_size = gemm_op.get_workspace_size(arguments);
    void *workspace = nullptr;
    if (workspace_size > 0) {
        cudaMallocAsync(&workspace, workspace_size, stream);
    }

    gemm_op.initialize(arguments, workspace, stream);
    gemm_op.run(stream);

    cudaFreeAsync(device_problem_shapes, stream);
    cudaFreeAsync(device_ptr_A, stream);
    cudaFreeAsync(device_ptr_B, stream);
    cudaFreeAsync(device_ptr_C, stream);
    cudaFreeAsync(device_ptr_D, stream);
    if (workspace) cudaFreeAsync(workspace, stream);

    delete[] host_segments;
    delete[] host_problem_shapes;
    delete[] host_ptr_A;
    delete[] host_ptr_B;
    delete[] host_ptr_C;
    delete[] host_ptr_D;
}
