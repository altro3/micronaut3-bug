cmake_minimum_required(VERSION 4.4.0)
project(cuda_kernel_tests LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include(FetchContent)

set(DOCTEST_VERSION "2.5.3")
set(DOCTEST_DOWNLOAD_URL "https://github.com/doctest/doctest/releases/download/v${DOCTEST_VERSION}/doctest-v${DOCTEST_VERSION}.tar.gz")
set(LOCAL_DOCTEST_DIR "D:/.my_llama_deps/doctest-v${DOCTEST_VERSION}")

if (NOT TARGET doctest)
    if (EXISTS "${LOCAL_DOCTEST_DIR}/CMakeLists.txt")
        message(STATUS "[MY_LLAMA] Found LOCAL doctest cache at ${LOCAL_DOCTEST_DIR}.")
        FetchContent_Declare(doctest SOURCE_DIR "${LOCAL_DOCTEST_DIR}")
    else ()
        message(STATUS "[MY_LLAMA] Local doctest not found. Downloading v${DOCTEST_VERSION} (.tar.gz)...")

        FetchContent_Declare(
                doctest_download
                URL "${DOCTEST_DOWNLOAD_URL}"
                DOWNLOAD_NO_EXTRACT TRUE
        )
        FetchContent_MakeAvailable(doctest_download)

        FetchContent_GetProperties(doctest_download SOURCE_DIR DOWNLOADED_DIR)
        set(ARCHIVE_PATH "${DOWNLOADED_DIR}/doctest-v${DOCTEST_VERSION}.tar.gz")

        file(MAKE_DIRECTORY "${LOCAL_DOCTEST_DIR}")
        execute_process(
                COMMAND tar -xf "${ARCHIVE_PATH}" -C "${LOCAL_DOCTEST_DIR}"
                RESULT_VARIABLE TAR_RESULT
        )

        if (NOT TAR_RESULT EQUAL 0)
            message(FATAL_ERROR "[MY_LLAMA] Native windows tar failed to extract doctest v${DOCTEST_VERSION}")
        endif ()
    endif ()

    FetchContent_Declare(doctest SOURCE_DIR "${LOCAL_DOCTEST_DIR}")
    FetchContent_MakeAvailable(doctest)
endif ()

file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/input_stage/*.cu"
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
        doctest::doctest
        cuda_kernels
        CUDA::cudart
        CUDA::cublas
)

enable_testing()
add_test(NAME ext_cuda_math_tests COMMAND run_cuda_tests)
