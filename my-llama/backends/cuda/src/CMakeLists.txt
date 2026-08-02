#set(CUDA_SOURCES
#        "${CMAKE_CURRENT_SOURCE_DIR}/input_stage/fused_multimodal_projection.cu"
#        "${CMAKE_CURRENT_SOURCE_DIR}/input_stage/fused_multimodal_projection_launcher.cu"
#        "${CMAKE_CURRENT_SOURCE_DIR}/input_stage/fused_rmsnorm_forward.cu"
#        "${CMAKE_CURRENT_SOURCE_DIR}/input_stage/fused_rmsnorm_forward_launcher.cu"
#        "${CMAKE_CURRENT_SOURCE_DIR}/input_stage/varlen_embeddings.cu"
#        "${CMAKE_CURRENT_SOURCE_DIR}/input_stage/varlen_embeddings_launcher.cu"
#
#        "${CMAKE_CURRENT_SOURCE_DIR}/prefill_stage/dequantize_fusion_mma_gguf_q4_k.cu"
#        "${CMAKE_CURRENT_SOURCE_DIR}/prefill_stage/dequantize_fusion_mma_gguf_q4_k_launcher.cu"
#        "${CMAKE_CURRENT_SOURCE_DIR}/prefill_stage/prefill_stage_dispatcher.cu"
#)

#[[
set(FLASHINFER_OBJ)
if (TARGET flashinfer_objects)
    set(FLASHINFER_OBJ $<TARGET_OBJECTS:flashinfer_objects>)
endif ()
]]

#[[
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
]]

#set_target_properties(pure_kernel_bench PROPERTIES
#        CXX_STANDARD 20
#        CUDA_STANDARD 20
#        CUDA_SEPARATE_COMPILATION OFF
#        CUDA_LINKER_LANGUAGE CUDA
#)

#target_compile_options(pure_kernel_bench PRIVATE
#        $<$<COMPILE_LANGUAGE:CUDA>:-DIN_PLACE_DEBUG>
#)
#
#target_include_directories(pure_kernel_bench PRIVATE
#        "${LOCAL_CUTLASS_DIR}/include"
#        "${LOCAL_CUTLASS_DIR}/tools/util/include"
#        "${CUDAToolkit_INCLUDE_DIRS}"
#)
#
#target_link_libraries(pure_kernel_bench PRIVATE
#        my_llama_cutlass_headers
#        cudart
#        cuda
#)

add_library(cuda_kernels SHARED
        prefill_stage/mini_tensor_test.cu
        prefill_stage/mini_tensor_test.cuh
)

target_link_libraries(cuda_kernels PRIVATE
        inference_compile_options
        cuda
)

target_include_directories(cuda_kernels PUBLIC
        "${LOCAL_CUTLASS_DIR}/include"
        "${LOCAL_CUTLASS_DIR}/tools/util/include"
        "${CUDAToolkit_INCLUDE_DIRS}"
        "${CUDAToolkit_INCLUDE_DIRS}/cccl"
)

target_compile_definitions(cuda_kernels PRIVATE EXPORT_KERNELS)

set_target_properties(cuda_kernels PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

#[[

set(CUDA_BIN_DIR "${CUDAToolkit_BIN_DIR}")
set(CUDA_INC_DIR "${CUDAToolkit_INCLUDE_DIRS}")

set(KERNEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/prefill_stage/mini_tensor_test.cu")
set(HEADER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/prefill_stage/mini_tensor_test.cuh")
set(TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/prefill_stage/main_test.cu")
set(EXE_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/pure_kernel_bench.exe")

add_custom_command(
        OUTPUT ${EXE_OUTPUT}
        COMMAND "${CUDA_BIN_DIR}/nvcc.exe"
        -std=c++20
        -O3
        -arch=sm_120
        -Xcompiler "/Zc:preprocessor"
        -Xcompiler "/MT"
        -I"${LOCAL_CUTLASS_DIR}/include"
        -I"${LOCAL_CUTLASS_DIR}/tools/util/include"
        -I"${CUDA_INC_DIR}"
        -I"${CUDA_INC_DIR}/cccl"
        -DIN_PLACE_DEBUG
        -Xlinker "/NODEFAULTLIB:cudart.lib"
        -Xlinker "/NODEFAULTLIB:msvcrt.lib"
        "${KERNEL_SRC}" "${TEST_SRC}"
        -o "${EXE_OUTPUT}"
        DEPENDS ${KERNEL_SRC} ${TEST_SRC} ${HEADER_SRC}
        COMMENT "[MY_LLAMA] Compiling and linking Blackwell CUTLASS kernel via strict NVCC pipeline..."
)

add_custom_target(pure_kernel_bench ALL DEPENDS ${EXE_OUTPUT})

set_target_properties(pure_kernel_bench PROPERTIES
        OUTPUT_NAME "pure_kernel_bench"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

add_executable(mini_tensor_test "${KERNEL_SRC}" "${TEST_SRC}" "${HEADER_SRC}")

set_target_properties(mini_tensor_test PROPERTIES
        CXX_STANDARD 20
        CUDA_STANDARD 20
        CUDA_SEPARATE_COMPILATION OFF
        CUDA_LINKER_LANGUAGE CUDA
)

target_include_directories(mini_tensor_test PRIVATE
        "${LOCAL_CUTLASS_DIR}/include"
        "${LOCAL_CUTLASS_DIR}/tools/util/include"
        "${CUDAToolkit_INCLUDE_DIRS}"
)

target_link_libraries(mini_tensor_test PRIVATE
        cudart_static
        cuda
)
]]
