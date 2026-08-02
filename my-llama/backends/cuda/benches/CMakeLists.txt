project(cuda_kernel_benches LANGUAGES CXX CUDA)

file(GLOB_RECURSE BENCH_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/input_stage/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/prefill_stage/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/bench_main.cpp"
)

add_executable(run_cuda_benches ${BENCH_SOURCES})

target_link_libraries(run_cuda_benches PRIVATE inference_compile_options)

if (MSVC)
    target_compile_options(run_cuda_benches PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/GS->
            $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    )
endif ()

target_include_directories(run_cuda_benches PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../include"
)

if (flashinfer_SOURCE_DIR)
    target_include_directories(run_cuda_benches PRIVATE "${flashinfer_SOURCE_DIR}/include")
endif ()

target_link_libraries(run_cuda_benches PRIVATE
        cuda_kernels
        cublas
)

set_target_properties(run_cuda_benches PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
