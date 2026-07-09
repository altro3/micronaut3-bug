cmake_minimum_required(VERSION 4.3)
project(my_llama_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES "100;90;89")

include_directories(cuda/include)
find_package(CUDAToolkit REQUIRED)

add_library(
        cuda_kernels STATIC
        cuda/src/adam.cu
#        cuda/src/argmax.cu
#        cuda/src/attention_scores.cu
#        cuda/src/attention_values.cu
#        cuda/src/embeddings.cu
#        cuda/src/linear_backward.cu
#        cuda/src/matmul.cu
#        cuda/src/rmsnorm.cu
#        cuda/src/softmax.cu
#        cuda/src/softmax_attention.cu
#        cuda/src/swiglu.cu
)

target_link_libraries(cuda_kernels PRIVATE CUDA::cublas CUDA::cublasLt)
