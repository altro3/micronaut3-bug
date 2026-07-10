cmake_minimum_required(VERSION 4.3)
project(my_llama_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES "100;90;89")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cuda/include)

find_package(CUDAToolkit REQUIRED)

add_library(
        cuda_kernels STATIC
        cuda/include/kernels.h
        cuda/src/adamw.cu
        cuda/src/dequantize_block_gguf_q4_k.cu
        cuda/src/fused_attention.cu
        cuda/src/matmul.cu
        cuda/src/matmul_gguf_q4_k.cu
        cuda/src/swiglu.cu
        cuda/src/rmsnorm.cu
)

target_link_libraries(cuda_kernels PRIVATE CUDA::cublas CUDA::cublasLt)
