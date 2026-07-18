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
    message(STATUS "[MY_LLAMA] Full Build Mode Active. Scanning folders for kernels...")

    # Собираем файлы .cu строго из наших целевых директорий бэкенда
    file(GLOB_RECURSE ATTENTION_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/attention/*.cu")
    file(GLOB_RECURSE DECODING_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/decoding/*.cu")
    file(GLOB_RECURSE LAYERS_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/layers/*.cu")
    file(GLOB_RECURSE QUANT_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/quantization/*.cu")
    file(GLOB_RECURSE SPEC_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/speculative/*.cu")
    file(GLOB_RECURSE UTILS_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cu")

    set(CUDA_SOURCES
            ${ATTENTION_SRCS}
            ${DECODING_SRCS}
            ${LAYERS_SRCS}
            ${QUANT_SRCS}
            ${SPEC_SRCS}
            ${UTILS_SRCS}
    )

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
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
)
