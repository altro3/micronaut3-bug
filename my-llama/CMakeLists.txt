cmake_minimum_required(VERSION 4.4.0)
project(my_llama_kernels_dispatcher LANGUAGES CXX)

set(BACKEND "CUDA" CACHE STRING "Target hardware backend: CUDA or MUSA")

if (BACKEND STREQUAL "CUDA")
    add_subdirectory(backends/cuda)
elseif (BACKEND STREQUAL "MUSA")
    add_subdirectory(backends/musa)
endif ()
