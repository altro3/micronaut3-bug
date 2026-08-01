set(CUDA_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/input_stage/fused_multimodal_projection.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/input_stage/fused_multimodal_projection_launcher.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/input_stage/fused_rmsnorm_forward.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/input_stage/fused_rmsnorm_forward_launcher.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/input_stage/varlen_embeddings.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/input_stage/varlen_embeddings_launcher.cu"

        "${CMAKE_CURRENT_SOURCE_DIR}/prefill_stage/dequantize_fusion_mma_gguf_q4_k.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/prefill_stage/dequantize_fusion_mma_gguf_q4_k_launcher.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/prefill_stage/prefill_stage_dispatcher.cu"
)

set(FLASHINFER_OBJ)
if (TARGET flashinfer_objects)
    set(FLASHINFER_OBJ $<TARGET_OBJECTS:flashinfer_objects>)
endif ()

add_library(cuda_kernels STATIC ${CUDA_SOURCES} ${FLASHINFER_OBJ})

target_include_directories(cuda_kernels PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../include"
        "${flashinfer_SOURCE_DIR}/include"
        "${LOCAL_CUTLASS_DIR}/include"
        "${LOCAL_CUTLASS_DIR}/tools/util/include"
        "${CUDAToolkit_INCLUDE_DIRS}"
)

set_target_properties(cuda_kernels PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_ARCHITECTURES "120"
        CUDA_SEPARABLE_COMPILATION OFF
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
)

target_link_libraries(cuda_kernels PRIVATE
        CUDA::cublas
        CUDA::cublasLt
        CUDA::cudart_static
)

add_executable(pure_kernel_bench "${CMAKE_CURRENT_SOURCE_DIR}/prefill_stage/mini_tensor_test.cu")

set_target_properties(pure_kernel_bench PROPERTIES
        CUDA_ARCHITECTURES "120"
        CXX_STANDARD 20
        CUDA_STANDARD 20
)


target_compile_options(pure_kernel_bench PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-DIN_PLACE_DEBUG --expt-relaxed-constexpr>
)

target_include_directories(pure_kernel_bench PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../include"
        "${LOCAL_CUTLASS_DIR}/include"
        "${LOCAL_CUTLASS_DIR}/tools/util/include"
        "${CUDAToolkit_INCLUDE_DIRS}"
)

target_link_libraries(pure_kernel_bench PRIVATE
        cuda_kernels
        CUDA::cudart_static
        CUDA::cublas
)