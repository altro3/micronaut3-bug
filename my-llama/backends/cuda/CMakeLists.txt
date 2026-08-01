cmake_minimum_required(VERSION 4.4.1)

set(CMAKE_CUDA_RUNTIME_LIBRARY "Static" CACHE STRING "CUDA runtime library" FORCE)

project(my_llama_cuda_backend LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CUTLASS_NVCC_ARCHS "120" CACHE STRING "CUDA architectures for CUTLASS kernels" FORCE)
set(CMAKE_CUDA_ARCHITECTURES "120" CACHE STRING "Standard CUDA architectures" FORCE)

set(FETCH_FLASHINFER ON CACHE BOOL "Fetch FlashInfer component")
set(FETCH_CUTLASS ON CACHE BOOL "Fetch CUTLASS component")

if (MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    add_compile_definitions(
            CUTLASS_ARCH_MMA_SM120_SUPPORTED=1
            CUTLASS_DEBUG_TRACE_LEVEL=0
            _CRT_SECURE_NO_WARNINGS
            NOMINMAX
    )
    foreach (flag_var
            CMAKE_CUDA_FLAGS CMAKE_CUDA_FLAGS_RELEASE CMAKE_CUDA_FLAGS_DEBUG
            CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG)
        set(${flag_var} "")
    endforeach ()
    set(CMAKE_CXX_FLAGS "/Zc:preprocessor /Zc:inline /Zc:lambda /Zc:__cplusplus /utf-8 /wd4068 /wd4624 /wd4190 /wd4018 /wd4244 /wd4267")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2")
    set(CMAKE_CXX_FLAGS_DEBUG "/Ob0 /Od")
else ()
    add_compile_definitions(
            CUTLASS_ARCH_MMA_SM120_SUPPORTED=1
            CUTLASS_DEBUG_TRACE_LEVEL=0
    )
endif ()

set(BLACKWELL_GENCODE "-gencode=arch=compute_120,code=sm_120")

if (MSVC)
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -Xptxas --maxrregcount=128 -Xptxas -v --diag-suppress=20012 -static-global-template-stub=false --expt-relaxed-constexpr -Xcompiler=/Zc:preprocessor -Xcompiler=/Zc:inline -Xcompiler=/Zc:lambda -Xcompiler=/MT")
    set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -Xptxas -O0 -Xcompiler=/Zc:preprocessor -Xcompiler=/Ob0 -Xcompiler=/Od -Xcompiler=/MTd")

    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -Xptxas --maxrregcount=128 -Xptxas -v --diag-suppress=20012 -static-global-template-stub=false --expt-relaxed-constexpr ${BLACKWELL_GENCODE} -Xcompiler=/Zc:preprocessor -Xcompiler=/Zc:inline -Xcompiler=/Zc:lambda")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2")
else ()
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -Xptxas --maxrregcount=128 -Xptxas -v --diag-suppress=20012 -static-global-template-stub=false --expt-relaxed-constexpr ${BLACKWELL_GENCODE}")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif ()

if (MSVC)
    set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -Xptxas -O0 ${BLACKWELL_GENCODE} -Xcompiler=/Zc:preprocessor -Xcompiler=/Ob0 -Xcompiler=/Od")
    set(CMAKE_CXX_FLAGS_DEBUG "/Ob0 /Od")
else ()
    set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -Xptxas -O0 ${BLACKWELL_GENCODE}")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0")
endif ()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Dependencies)

add_subdirectory(src)

option(BUILD_TESTS "Build C++ CUDA kernel tests" ON)
if (BUILD_TESTS AND NOT ISOLATED_BUILD)
    enable_testing()
    add_subdirectory(tests)
endif ()
