cmake_minimum_required(VERSION 4.4.0)
project(my_llama_cuda_backend LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES "120")
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

if (MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/std:c++20>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)

    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /MACHINE:X64")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /MACHINE:X64")
endif ()

set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -Xptxas --maxrregcount=128 -Xptxas -O1 -Xptxas -v --diag-suppress=20012 -static-global-template-stub=false")
if (MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /GL")
else ()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif ()

set(CMAKE_CUDA_FLAGS_DEBUG "-g -O0 -Xptxas -O0")
if (MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "/Zi /Ob0 /Od")
else ()
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
endif ()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Dependencies)

add_subdirectory(src)

option(BUILD_TESTS "Build C++ CUDA kernel tests" ON)
if (BUILD_TESTS AND NOT ISOLATED_BUILD)
    enable_testing()
    add_subdirectory(tests)
endif ()
