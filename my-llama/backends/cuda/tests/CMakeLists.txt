project(cuda_kernel_tests LANGUAGES CXX CUDA)

include(FetchContent)

set(DOCTEST_VERSION "2.5.3")
download_llama_dependency(doctest ${DOCTEST_VERSION} "https://github.com/doctest/doctest/releases/download/v${DOCTEST_VERSION}/doctest-v${DOCTEST_VERSION}.tar.gz" "")

file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/input_stage/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/prefill_stage/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_main.cpp"
)

add_executable(run_cuda_tests ${TEST_SOURCES})

if (MSVC)
    target_compile_options(run_cuda_tests PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/GS->
    )
endif ()

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
add_test(NAME ext_cuda_math_tests COMMAND run_cuda_tests -s)
