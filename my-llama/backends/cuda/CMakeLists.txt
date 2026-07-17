cmake_minimum_required(VERSION 4.4.0)
project(my_llama_cuda_backend LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES "120")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -static-global-template-stub=false")
set(CMAKE_CXX_FLAGS_RELEASE "/O2")

set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -static-global-template-stub=false")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

include(FetchContent)

set(NT_HOST "https://github.com")
set(NT_PATH "NVIDIA/cutlass.git")
string(CONCAT CUTLASS_GIT_URL ${NT_HOST} "/" ${NT_PATH})

FetchContent_Declare(
        cutlass
        GIT_REPOSITORY "${CUTLASS_GIT_URL}"
        GIT_TAG "v4.6.1"
        GIT_SHALLOW ON
        SOURCE_SUBDIR "non_existent_directory"
)
FetchContent_MakeAvailable(cutlass)

include_directories(${cutlass_SOURCE_DIR}/include)

find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")

add_library(cuda_kernels STATIC ${CUDA_SOURCES})
target_link_libraries(cuda_kernels PRIVATE CUDA::cublas CUDA::cublasLt)
