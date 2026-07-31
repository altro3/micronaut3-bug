cmake_minimum_required(VERSION 4.4.1)
project(cuda_kernel_tests LANGUAGES CXX CUDA)

include(FetchContent)

set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CUDA_ARCHITECTURES "120")
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(DOCTEST_VERSION "2.5.3")
FetchContent_Declare(
        doctest
        GIT_REPOSITORY https://github.com
        GIT_TAG "v${DOCTEST_VERSION}"
)
FetchContent_MakeAvailable(doctest)

file(GLOB_RECURSE TEST_FILES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/input_stage/test_*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/prefill_stage/test_*.cu"
)

set(CUDA_COMMON_INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
        ${flashinfer_SOURCE_DIR}/include
        ${LOCAL_CUTLASS_DIR}/include
        ${LOCAL_CUTLASS_DIR}/tools/util/include
        ${CUDAToolkit_INCLUDE_DIRS}
)

foreach (test_file ${TEST_FILES})
    get_filename_component(test_name ${test_file} NAME_WE)

    add_executable(${test_name} ${test_file} "${CMAKE_CURRENT_SOURCE_DIR}/test_main.cpp")

    if (MSVC)
        target_compile_options(${test_name} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/GS->
                $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_120,code=sm_120>
        )
    else ()
        target_compile_options(${test_name} PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_120,code=sm_120>
        )
    endif ()

    target_compile_options(${test_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -O3 --use_fast_math -Xptxas -v>
    )

    target_include_directories(${test_name} PRIVATE
            ${LOCAL_CUTLASS_DIR}/include
            ${LOCAL_CUTLASS_DIR}/tools/util/include
            ${CUDA_COMMON_INCLUDE_DIRS}
    )

    target_link_libraries(${test_name} PRIVATE
            doctest::doctest
            cuda_kernels
            CUDA::cudart
            CUDA::cublas
    )

    set_target_properties(${test_name} PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )

    add_test(NAME ${test_name} COMMAND ${test_name} -s)
endforeach ()