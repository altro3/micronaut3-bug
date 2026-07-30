include(FetchContent)

set(DOCTEST_VERSION "2.5.3")
download_llama_dependency(doctest ${DOCTEST_VERSION} "https://github.com/doctest/doctest/releases/download/v${DOCTEST_VERSION}/doctest-v${DOCTEST_VERSION}.tar.gz" "")

file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/input_stage/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/prefill_stage/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_main.cpp"
)

add_executable(run_cuda_tests ${TEST_SOURCES})

set_target_properties(run_cuda_tests PROPERTIES
        CUDA_ARCHITECTURES "120"
)

if (MSVC)
    target_compile_options(run_cuda_tests PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/GS->
            $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_120,code=sm_120>
    )
else ()
    target_compile_options(run_cuda_tests PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_120,code=sm_120>
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
