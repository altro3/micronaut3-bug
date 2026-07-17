cmake_minimum_required(VERSION 4.4.0)
project(my_llama_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES "100;90;89")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -O3")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cuda/include)

include(FetchContent)

set(NT_HOST "https://github.com")
set(NT_PATH "NVIDIA/cutlass.git")
string(CONCAT CUTLASS_GIT_URL ${NT_HOST} "/" ${NT_PATH})

FetchContent_Declare(
        cutlass
        GIT_REPOSITORY "${CUTLASS_GIT_URL}"
        GIT_TAG "v4.6.1"
        GIT_SHALLOW ON
)
FetchContent_MakeAvailable(cutlass)

include_directories(${cutlass_SOURCE_DIR}/include)

find_package(CUDAToolkit REQUIRED)

file(GLOB CUDA_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cuda/src/*.cu")
add_library(cuda_kernels STATIC ${CUDA_SOURCES})

target_link_libraries(cuda_kernels PRIVATE CUDA::cublas CUDA::cublasLt)
