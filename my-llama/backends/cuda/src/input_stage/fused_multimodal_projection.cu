#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/callbacks.hpp"
#include "cutlass/epilogue/fusion/sm120_visitor_store_tma_warpspecialized.hpp"

using namespace cute;

extern "C" void launch_fused_multimodal_projection(
    void* __restrict__ out_tokens,
    const void* __restrict__ input_tokens,
    const void* __restrict__ weight_matrix,
    const float* __restrict__ bias,
    const int32_t* __restrict__ vision_segments,
    int32_t num_segments,
    int32_t vision_hidden_size,
    int32_t text_hidden_size,
    int32_t tp_rank,
    int32_t tp_size,
    cudaStream_t stream
) {
    const int32_t local_out_features = text_hidden_size / tp_size;

    using ElementA = bfloat16_t;
    using LayoutA  = cutlass::layout::RowMajor;
    using ElementB = bfloat16_t;
    using LayoutB  = cutlass::layout::ColumnMajor;
    using ElementC = bfloat16_t;
    using LayoutC  = cutlass::layout::RowMajor;

    using ElementAccumulator = float;
    using ElementCompute     = float;

    using ArchTag            = cutlass::arch::Sm120;
    using OperatorClass      = cutlass::arch::OpClassTensorOp;

    using TileShape = Shape<_128, _128, _64>;
    using ClusterShape = Shape<_1, _1, _1>;

    using EpilogueDescriptor = cutlass::epilogue::collective::detail::EpilogueDescriptor<
        TileShape,
        ClusterShape,
        ElementC,
        ElementC,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >;

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

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag,                         // 1. ArchTag (Sm120)
        OperatorClass,                   // 2. OpClassTensorOp
        TileShape,                       // 3. TileShape
        ClusterShape,                    // 4. ClusterShape
        cutlass::epilogue::collective::EpilogueTileAuto, // 5. EpilogueTileType
        ElementAccumulator,              // 6. ElementAccumulator (float)
        ElementCompute,                  // 7. ElementCompute (float)
        ElementC,                        // 8. ElementC (bfloat16_t)
        LayoutC,                         // 9. GmemLayoutTagC (RowMajor)
        4,                               // 10. AlignmentC
        ElementC,                        // 11. ElementD (bfloat16_t)
        LayoutC,                         // 12. GmemLayoutTagD (RowMajor)
        4,                               // 13. AlignmentD
        cutlass::epilogue::collective::EpilogueScheduleAuto, // 14. EpilogueScheduleType
        EVT_BiasGELU                     // 15. FusionOpOrCallbacks
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120,
        cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, 4,
        ElementB, LayoutB, 4,
        ElementAccumulator,
        TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAuto,
        cutlass::gemm::KernelTmaWarpSpecializedCooperativeSm120<1> // Добавили строго <1>
    >::CollectiveOp;


    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue
    >;

    using GemmGroupedUniversal = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    int32_t* host_segments = new int32_t[num_segments * 2];
    cudaMemcpyAsync(host_segments, vision_segments, num_segments * 2 * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    using ProblemShapeType = Shape<int, int, int>;
    ProblemShapeType* host_problem_shapes = new ProblemShapeType[num_segments];

    ElementA** host_ptr_A = new ElementA*[num_segments];
    ElementB** host_ptr_B = new ElementB*[num_segments];
    ElementC** host_ptr_C = new ElementC*[num_segments];
    ElementC** host_ptr_D = new ElementC*[num_segments];
    float** host_ptr_Bias = new float*[num_segments];

    for (int32_t i = 0; i < num_segments; ++i) {
        int32_t start_token = host_segments[i * 2];
        int32_t end_token = host_segments[i * 2 + 1];
        int32_t segment_tokens = end_token - start_token;

        host_problem_shapes[i] = make_shape(segment_tokens, local_out_features, vision_hidden_size);

        host_ptr_A[i] = const_cast<ElementA*>(reinterpret_cast<const ElementA*>(input_tokens)) + start_token * vision_hidden_size;
        host_ptr_B[i] = const_cast<ElementB*>(reinterpret_cast<const ElementB*>(weight_matrix));
        host_ptr_C[i] = reinterpret_cast<ElementC*>(out_tokens) + start_token * local_out_features;
        host_ptr_D[i] = reinterpret_cast<ElementC*>(out_tokens) + start_token * local_out_features;
        host_ptr_Bias[i] = const_cast<float*>(bias);
    }

    ProblemShapeType* device_problem_shapes;
    ElementA** device_ptr_A;
    ElementB** device_ptr_B;
    ElementC** device_ptr_C;
    ElementC** device_ptr_D;
    float** device_ptr_Bias;

    cudaMalloc(&device_problem_shapes, num_segments * sizeof(ProblemShapeType));
    cudaMalloc(&device_ptr_A, num_segments * sizeof(ElementA*));
    cudaMalloc(&device_ptr_B, num_segments * sizeof(ElementB*));
    cudaMalloc(&device_ptr_C, num_segments * sizeof(ElementC*));
    cudaMalloc(&device_ptr_D, num_segments * sizeof(ElementC*));
    cudaMalloc(&device_ptr_Bias, num_segments * sizeof(float*));

    cudaMemcpyAsync(device_problem_shapes, host_problem_shapes, num_segments * sizeof(ProblemShapeType), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_ptr_A, host_ptr_A, num_segments * sizeof(ElementA*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_ptr_B, host_ptr_B, num_segments * sizeof(ElementB*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_ptr_C, host_ptr_C, num_segments * sizeof(ElementC*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_ptr_D, host_ptr_D, num_segments * sizeof(ElementC*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_ptr_Bias, host_ptr_Bias, num_segments * sizeof(float*), cudaMemcpyHostToDevice, stream);

    auto stride_A = make_stride(vision_hidden_size, _1{});
    auto stride_B = make_stride(_1{}, vision_hidden_size);
    auto stride_C = make_stride(local_out_features, _1{});
    auto stride_D = make_stride(local_out_features, _1{});

    typename GemmGroupedUniversal::Arguments arguments;
    arguments.mode = cutlass::gemm::GemmUniversalMode::kGrouped;
    arguments.problem_shape.problem_shapes = device_problem_shapes;
    arguments.problem_shape.problem_count = num_segments;

    arguments.mainloop.ptr_A = device_ptr_A;
    arguments.mainloop.stride_A = stride_A;
    arguments.mainloop.ptr_B = device_ptr_B;
    arguments.mainloop.stride_B = stride_B;

    arguments.epilogue.thread_args.template get<0>().template get<1>().ptr_B = device_ptr_Bias;
    arguments.epilogue.ptr_C = device_ptr_C;
    arguments.epilogue.stride_C = stride_C;
    arguments.epilogue.ptr_D = device_ptr_D;
    arguments.epilogue.stride_D = stride_D;

    GemmGroupedUniversal gemm_op;

    size_t workspace_size = gemm_op.get_workspace_size(arguments);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    gemm_op.initialize(arguments, workspace, stream);
    gemm_op.run(stream);

    cudaFree(device_problem_shapes);
    cudaFree(device_ptr_A);
    cudaFree(device_ptr_B);
    cudaFree(device_ptr_C);
    cudaFree(device_ptr_D);
    cudaFree(device_ptr_Bias);
    if (workspace) cudaFree(workspace);

    delete[] host_segments;
    delete[] host_problem_shapes;
    delete[] host_ptr_A;
    delete[] host_ptr_B;
    delete[] host_ptr_C;
    delete[] host_ptr_D;
    delete[] host_ptr_Bias;
}
