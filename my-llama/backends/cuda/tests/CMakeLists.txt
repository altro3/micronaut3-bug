cmake_minimum_required(VERSION 4.4.0)
project(cuda_kernel_tests LANGUAGES CXX CUDA)

if (NOT TARGET gtest_main)
    set(LOCAL_GTEST_DIR "D:/.my_llama_deps/googletest-main")
    if (EXISTS "${LOCAL_GTEST_DIR}/CMakeLists.txt")
        message(STATUS "[MY_LLAMA] Found LOCAL GoogleTest cache at ${LOCAL_GTEST_DIR}.")
        FetchContent_Declare(googletest SOURCE_DIR "${LOCAL_GTEST_DIR}")
    else ()
        message(STATUS "[MY_LLAMA] Local GoogleTest not found. Downloading...")
        FetchContent_Declare(
                googletest
                URL "${NT_HOST}/google/googletest/archive/refs/heads/main.zip"
                SOURCE_DIR "${LOCAL_GTEST_DIR}"
        )
    endif ()
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif ()

file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
        #        "${CMAKE_CURRENT_SOURCE_DIR}/input_stage/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/prefill_stage/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_main.cpp"
)

add_executable(run_cuda_tests ${TEST_SOURCES})

target_include_directories(run_cuda_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
        ${flashinfer_SOURCE_DIR}/include
        ${LOCAL_CUTLASS_DIR}/include
        ${LOCAL_CUTLASS_DIR}/tools/util/include
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(run_cuda_tests PRIVATE
        gtest_main
        cuda_kernels
        CUDA::cudart
        CUDA::cublas
)

include(GoogleTest)
gtest_discover_tests(run_cuda_tests)
