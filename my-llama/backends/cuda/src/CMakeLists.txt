set(CUDA_SOURCES
        "input_stage/fused_multimodal_projection.cu"
        "input_stage/fused_multimodal_projection_launcher.cu"
        "input_stage/fused_rmsnorm_forward.cu"
        "input_stage/fused_rmsnorm_forward_launcher.cu"
        "input_stage/varlen_embeddings.cu"
        "input_stage/varlen_embeddings_launcher.cu"
        "prefill_stage/dequantize_fusion_mma_gguf_q4_k.cu"
        "prefill_stage/dequantize_fusion_mma_gguf_q4_k_launcher.cu"
        "prefill_stage/prefill_stage_dispatcher.cu"
)

set(FLASHINFER_OBJ)
if (TARGET flashinfer_objects)
    set(FLASHINFER_OBJ $<TARGET_OBJECTS:flashinfer_objects>)
endif ()

add_library(cuda_kernels STATIC ${CUDA_SOURCES} ${FLASHINFER_OBJ})

target_include_directories(cuda_kernels PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${flashinfer_SOURCE_DIR}/include
        ${LOCAL_CUTLASS_DIR}/include
        ${LOCAL_CUTLASS_DIR}/tools/util/include
        ${CUDAToolkit_INCLUDE_DIRS}
)

set_target_properties(cuda_kernels PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_ARCHITECTURES "120"
        CUDA_SEPARABLE_COMPILATION OFF
)

target_link_libraries(cuda_kernels PRIVATE
        CUDA::cublas
        CUDA::cublasLt
        CUDA::cudart_static
)
