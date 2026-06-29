cmake_minimum_required(VERSION 4.3)
project(my_qwen_cuda LANGUAGES CXX CUDA)

# Говорим плагину, где искать наши заголовочные файлы
include_directories(cuda/include)

# Перечисляем Си-файлы, чтобы IDEA объединила их в один проект
add_library(cuda_kernels STATIC
    cuda/src/matmul.cu
    cuda/src/softmax.cu
    cuda/src/attention.cu
    cuda/src/adam.cu
)
