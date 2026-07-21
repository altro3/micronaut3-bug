cmake_minimum_required(VERSION 4.4.0)
project(my_llama_cuda_backend LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

set(CMAKE_CUDA_ARCHITECTURES "120")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -Xptxas -v --diag-suppress=20012 -static-global-template-stub=false")
set(CMAKE_CXX_FLAGS_RELEASE "/O2 /GL /Zc:preprocessor")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Dependencies)

add_subdirectory(src)
