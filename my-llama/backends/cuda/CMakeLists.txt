cmake_minimum_required(VERSION 4.4.0)
project(my_llama_cuda_backend LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if (MSVC)
    string(APPEND CMAKE_CXX_FLAGS " /Zc:preprocessor")
    string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler=/Zc:preprocessor")
    include(Platform/Windows-MSVC)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /MACHINE:X64")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /MACHINE:X64")
endif ()

set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

set(CMAKE_CUDA_ARCHITECTURES "120")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -Xptxas -v --diag-suppress=20012 -static-global-template-stub=false")
set(CMAKE_CXX_FLAGS_RELEASE "/O2 /GL /Zc:preprocessor")

set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0 -Xptxas -O0")
if (MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "/MDd /Zi /Ob0 /Od /RTC1")
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
