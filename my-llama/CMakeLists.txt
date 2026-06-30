cmake_minimum_required(VERSION 4.3)
project(my_llama_cuda LANGUAGES CXX CUDA)

include_directories(cuda/include)

add_library(
        cuda_kernels STATIC
        cuda/src/matmul.cu
        cuda/src/swiglu.cu
        cuda/src/linear_backward.cu
        cuda/src/argmax.cu
        cuda/src/rmsnorm.cu
        cuda/src/softmax.cu
        cuda/src/attention.cu
        cuda/src/adam.cu
)
