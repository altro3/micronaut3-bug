project(cuda_kernel_benches LANGUAGES CXX CUDA)

file(GLOB_RECURSE BENCH_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/input_stage/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/prefill_stage/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/bench_main.cpp"
)

add_executable(run_cuda_benches ${BENCH_SOURCES})

if (MSVC)
    target_compile_options(run_cuda_benches PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/GS->
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/std:c++20>
            $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    )
endif ()

target_include_directories(run_cuda_benches PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
        ${flashinfer_SOURCE_DIR}/include
        ${LOCAL_CUTLASS_DIR}/include
        ${LOCAL_CUTLASS_DIR}/tools/util/include
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(run_cuda_benches PRIVATE
        cuda_kernels
        CUDA::cudart
        CUDA::cublas
)
