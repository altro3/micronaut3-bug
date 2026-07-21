include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
include_directories(${flashinfer_SOURCE_DIR}/include)
include_directories(${LOCAL_CUTLASS_DIR}/include)
include_directories(${LOCAL_CUTLASS_DIR}/tools/util/include)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

set(FLASHINFER_OBJ)
if(TARGET flashinfer_objects)
    set(FLASHINFER_OBJ $<TARGET_OBJECTS:flashinfer_objects>)
endif()

if(ISOLATED_BUILD)
    message(STATUS "[MY_LLAMA] Isolated Mode Active.")
    set(DUMMY_FILE "${CMAKE_CURRENT_BINARY_DIR}/dummy_stub.cpp")
    if(NOT EXISTS "${DUMMY_FILE}")
        file(WRITE "${DUMMY_FILE}" "// Stub.\n")
    endif()
    add_library(cuda_kernels STATIC "${DUMMY_FILE}" ${FLASHINFER_OBJ})
else()
    message(STATUS "[MY_LLAMA] Full Build Mode Active.")

    set(INPUT_STAGE_SRCS
            "${CMAKE_CURRENT_SOURCE_DIR}/input_stage/fused_multimodal_projection.cu"
            "${CMAKE_CURRENT_SOURCE_DIR}/input_stage/fused_multimodal_projection_launcher.cu"
            "${CMAKE_CURRENT_SOURCE_DIR}/input_stage/fused_rmsnorm_forward.cu"
            "${CMAKE_CURRENT_SOURCE_DIR}/input_stage/fused_rmsnorm_forward_launcher.cu"
            "${CMAKE_CURRENT_SOURCE_DIR}/input_stage/varlen_embeddings.cu"
            "${CMAKE_CURRENT_SOURCE_DIR}/input_stage/varlen_embeddings_launcher.cu"
    )

    file(GLOB_RECURSE ATTENTION_SRCS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/attention/*.cu" "${CMAKE_CURRENT_SOURCE_DIR}/attention/*.cuh")
    file(GLOB_RECURSE DECODING_SRCS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/decoding/*.cu" "${CMAKE_CURRENT_SOURCE_DIR}/decoding/*.cuh")
    file(GLOB_RECURSE LAYERS_SRCS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/layers/*.cu"  "${CMAKE_CURRENT_SOURCE_DIR}/layers/*.cuh")
    file(GLOB_RECURSE QUANT_SRCS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/quantization/*.cu" "${CMAKE_CURRENT_SOURCE_DIR}/quantization/*.cuh")
    file(GLOB_RECURSE SPEC_SRCS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/speculative/*.cu" "${CMAKE_CURRENT_SOURCE_DIR}/speculative/*.cuh")
    file(GLOB_RECURSE UTILS_SRCS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cu" "${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cuh")
    file(GLOB GLOBAL_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../include/*.h")

    set(CUDA_SOURCES
            ${ATTENTION_SRCS}
            ${DECODING_SRCS}
            ${INPUT_STAGE_SRCS}
            ${LAYERS_SRCS}
            ${QUANT_SRCS}
            ${SPEC_SRCS}
            ${UTILS_SRCS}
            ${GLOBAL_HEADERS}
    )

    add_library(cuda_kernels STATIC ${CUDA_SOURCES} ${FLASHINFER_OBJ})
endif()

set_target_properties(cuda_kernels PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)
