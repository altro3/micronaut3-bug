cmake_minimum_required(VERSION 4.3)
project(my_llama_cuda LANGUAGES CXX CUDA)

include_directories(cuda/include)

add_library(
        cuda_kernels STATIC
        cuda/src/matmul.cu
        cuda/src/softmax.cu
        cuda/src/attention.cu
        cuda/src/adam.cu
)
