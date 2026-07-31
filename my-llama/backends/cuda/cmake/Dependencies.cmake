include(FetchContent)
find_package(CUDAToolkit REQUIRED)

set(MY_LLAMA_DEPS_DIR "D:/.my_llama_deps" CACHE PATH "Directory for localized CUDA/AI dependencies cache")

macro(download_llama_dependency DEP_NAME DEP_SOURCE DEP_TAG)
    set(LOCAL_DEP_DIR "${MY_LLAMA_DEPS_DIR}/${DEP_NAME}-${DEP_TAG}")

    if (NOT TARGET ${DEP_NAME})
        if (EXISTS "${LOCAL_DEP_DIR}/include" OR EXISTS "${LOCAL_DEP_DIR}/CMakeLists.txt")
            message(STATUS "[MY_LLAMA] Found LOCAL ${DEP_NAME} cache at ${LOCAL_DEP_DIR}. Skipping network download!")
            FetchContent_Declare(${DEP_NAME} SOURCE_DIR "${LOCAL_DEP_DIR}")
        else ()
            message(STATUS "[MY_LLAMA] ${DEP_NAME} not found locally. Fetching...")

            if ("${DEP_SOURCE}" MATCHES "\\.git$")
                FetchContent_Declare(
                        ${DEP_NAME}_download
                        GIT_REPOSITORY "${DEP_SOURCE}"
                        GIT_TAG "${DEP_TAG}"
                        GIT_SHALLOW TRUE
                )
            else ()
                FetchContent_Declare(${DEP_NAME}_download URL "${DEP_SOURCE}" DOWNLOAD_NO_EXTRACT TRUE)
            endif ()

            FetchContent_MakeAvailable(${DEP_NAME}_download)

            FetchContent_GetProperties(${DEP_NAME}_download SOURCE_DIR DOWNLOADED_DIR)
            file(MAKE_DIRECTORY "${LOCAL_DEP_DIR}")

            if ("${DEP_SOURCE}" MATCHES "\\.git$")
                file(COPY ${DOWNLOADED_DIR}/ DESTINATION "${LOCAL_DEP_DIR}")
            else ()
                file(GLOB ARCHIVE_FILE "${DOWNLOADED_DIR}/*.tar.gz" "${DOWNLOADED_DIR}/*.zip")
                execute_process(
                        COMMAND tar -xf "${ARCHIVE_FILE}" -C "${LOCAL_DEP_DIR}" --strip-components=1
                        RESULT_VARIABLE TAR_RESULT
                )
                if (NOT TAR_RESULT EQUAL 0)
                    message(FATAL_ERROR "[MY_LLAMA] Native windows tar failed to extract ${DEP_NAME} ${DEP_TAG}")
                endif ()
            endif ()
        endif ()

        FetchContent_Declare(${DEP_NAME} SOURCE_DIR "${LOCAL_DEP_DIR}")
        FetchContent_MakeAvailable(${DEP_NAME})
    endif ()
endmacro()

if (NOT DEFINED FLASHINFER_TAG)
    set(FLASHINFER_TAG "v0.6.15")
endif ()

if (NOT DEFINED CUTLASS_TAG)
    set(CUTLASS_TAG "v4.6.1")
endif ()

set(LOCAL_FLASHINFER_DIR "${MY_LLAMA_DEPS_DIR}/flashinfer-${FLASHINFER_TAG}")
set(LOCAL_CUTLASS_DIR "${MY_LLAMA_DEPS_DIR}/cutlass-${CUTLASS_TAG}")

download_llama_dependency(
        flashinfer
        "https://github.com/flashinfer-ai/flashinfer/archive/refs/tags/${FLASHINFER_TAG}.zip"
        "${FLASHINFER_TAG}"
)
set(flashinfer_SOURCE_DIR "${LOCAL_FLASHINFER_DIR}")

set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable only CUTLASS headers" FORCE)
set(CUTLASS_ENABLE_EXAMPLES OFF CACHE BOOL "Disable CUTLASS examples" FORCE)
set(CUTLASS_ENABLE_TESTS OFF CACHE BOOL "Disable CUTLASS tests" FORCE)

download_llama_dependency(
        cutlass
        "https://github.com//NVIDIA/cutlass/archive/refs/tags/${CUTLASS_TAG}.zip"
        "${CUTLASS_TAG}"
)
