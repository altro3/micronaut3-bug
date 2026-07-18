include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
include_directories(${flashinfer_SOURCE_DIR}/include)
include_directories(${cutlass_SOURCE_DIR}/include)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

set(FLASHINFER_OBJ)
if(TARGET flashinfer_objects)
    set(FLASHINFER_OBJ $<TARGET_OBJECTS:flashinfer_objects>)
endif()

if(ISOLATED_BUILD)
    message(STATUS "[MY_LLAMA] Isolated Mode Active. Compiling only third-party objects.")
    set(DUMMY_FILE "${CMAKE_CURRENT_BINARY_DIR}/dummy_stub.cpp")
    if(NOT EXISTS "${DUMMY_FILE}")
        file(WRITE "${DUMMY_FILE}" "// Born to satisfy MSBuild.\n")
    endif()
    add_library(cuda_kernels STATIC "${DUMMY_FILE}" ${FLASHINFER_OBJ})
    target_link_libraries(cuda_kernels PRIVATE cudart.lib)
else()
    message(STATUS "[MY_LLAMA] Full Build Mode Active. Scanning folders with CONFIGURE_DEPENDS...")

    file(GLOB_RECURSE ATTENTION_SRCS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/attention/*.cu" "${CMAKE_CURRENT_SOURCE_DIR}/attention/*.cuh")
    file(GLOB_RECURSE DECODING_SRCS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/decoding/*.cu" "${CMAKE_CURRENT_SOURCE_DIR}/decoding/*.cuh")
    file(GLOB_RECURSE INPUT_STAGE_SRCS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/input_stage/*.cu" "${CMAKE_CURRENT_SOURCE_DIR}/input_stage/*.cuh")
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

    set(CMAKE_CUDA_SEPARATE_COMPILATION ON)

    add_library(cuda_kernels STATIC ${CUDA_SOURCES} ${FLASHINFER_OBJ})

    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(cuda_kernels
            PRIVATE
            CUDA::cublas
            CUDA::cublasLt
            CUDA::cudart
    )
endif()

set_target_properties(cuda_kernels PROPERTIES
        CUDA_SEPARATE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
)
