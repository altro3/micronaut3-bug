cmake_minimum_required(VERSION 4.3)
project(my_llama_cuda LANGUAGES CXX CUDA)

include_directories(cuda/include)

find_package(CUDAToolkit REQUIRED)

add_library(
        cuda_kernels STATIC
        cuda/src/matmul.cu
        cuda/src/swiglu.cu
        cuda/src/linear_backward.cu
        cuda/src/argmax.cu
        cuda/src/rmsnorm.cu
        cuda/src/softmax.cu
        cuda/src/attention_scores.cu
        cuda/src/softmax_attention.cu
        cuda/src/attention_values.cu
        cuda/src/adam.cu
)

target_link_libraries(cuda_kernels PRIVATE CUDA::cublas CUDA::cublasLt)
