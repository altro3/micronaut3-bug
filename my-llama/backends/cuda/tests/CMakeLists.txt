set(DOCTEST_TAG "v2.5.3")

download_llama_dependency(
        doctest
        "https://github.com/doctest/doctest/releases/download/${DOCTEST_TAG}/doctest-${DOCTEST_TAG}.tar.gz"
        "${DOCTEST_TAG}"
)

if (NOT TARGET doctest::doctest)
    add_library(doctest::doctest INTERFACE IMPORTED GLOBALLY)
    set_target_properties(doctest::doctest PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${MY_LLAMA_DEPS_DIR}/doctest-${DOCTEST_TAG}"
    )
endif ()

file(GLOB_RECURSE TEST_FILES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/input_stage/test_*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/prefill_stage/test_*.cu"
)

set(TESTS_SPECIFIC_INCLUDES
        "${CMAKE_CURRENT_SOURCE_DIR}/../include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../src"
)

if (flashinfer_SOURCE_DIR)
    list(APPEND TESTS_SPECIFIC_INCLUDES "${flashinfer_SOURCE_DIR}/include")
endif ()

set(ALL_TEST_EXECUTABLES "")

if (MSVC)
    file(TO_CMAKE_PATH "$ENV{LIB}" MSVC_LIB_PATHS)
    foreach (path ${MSVC_LIB_PATHS})
        link_directories("${path}")
    endforeach ()
endif ()

foreach (test_file ${TEST_FILES})
    get_filename_component(test_name ${test_file} NAME_WE)

    add_executable(${test_name} ${test_file} "${CMAKE_CURRENT_SOURCE_DIR}/test_main.cpp")
    set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/test_main.cpp" PROPERTIES
            LANGUAGE CUDA
    )

    list(APPEND ALL_TEST_EXECUTABLES ${test_name})

    set_target_properties(${test_name} PROPERTIES
            CXX_SCAN_FOR_MODULES OFF
            CUDA_SEPARABLE_COMPILATION OFF
            RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin"
            RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    target_link_libraries(${test_name} PRIVATE inference_compile_options)

    if (MSVC)
        target_compile_options(${test_name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/GS->)
    endif ()

    target_compile_options(${test_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -Xptxas -v>)
    target_include_directories(${test_name} PRIVATE ${TESTS_SPECIFIC_INCLUDES})
    add_dependencies(${test_name} cuda_kernels)

    target_link_libraries(${test_name} PRIVATE
            doctest::doctest
            cuda_kernels
            cublas
    )

    add_test(NAME ${test_name} COMMAND ${test_name} -s)
    set_tests_properties(${test_name} PROPERTIES ENVIRONMENT "PATH=${CMAKE_BINARY_DIR}/bin\;$ENV{PATH}")
endforeach ()

if (ALL_TEST_EXECUTABLES)
    add_custom_target(run_tests
            COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --output-on-failure
            COMMENT "[MY_LLAMA] Running all dynamic inference engine tests via CTest..."
    )

    add_dependencies(run_tests ${ALL_TEST_EXECUTABLES})
endif ()
