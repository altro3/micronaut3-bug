include(FetchContent)
set(NT_HOST "https://github.com")

find_package(CUDAToolkit REQUIRED)

if (NOT DEFINED FLASHINFER_TAG)
    set(FLASHINFER_TAG "v0.6.15")
endif ()

if (NOT DEFINED CUTLASS_TAG)
    set(CUTLASS_TAG "v4.6.1")
endif ()

set(LOCAL_FLASHINFER_DIR "D:/.my_llama_deps/flashinfer-${FLASHINFER_TAG}")
set(LOCAL_CUTLASS_DIR "D:/.my_llama_deps/cutlass-${CUTLASS_TAG}")

if (FETCH_FLASHINFER)
    if (EXISTS "${LOCAL_FLASHINFER_DIR}/include")
        message(STATUS "[MY_LLAMA] Found LOCAL FlashInfer cache (${FLASHINFER_TAG}) at ${LOCAL_FLASHINFER_DIR}. Skipping network download!")
        FetchContent_Declare(
                flashinfer_ext
                SOURCE_DIR "${LOCAL_FLASHINFER_DIR}"
        )
    else ()
        message(STATUS "[MY_LLAMA] Local FlashInfer (${FLASHINFER_TAG}) not found. Fetching via network...")
        FetchContent_Declare(
                flashinfer_ext
                URL "${NT_HOST}/flashinfer-ai/flashinfer/archive/refs/tags/${FLASHINFER_TAG}.zip"
                SOURCE_DIR "${LOCAL_FLASHINFER_DIR}"
        )
    endif ()
    FetchContent_MakeAvailable(flashinfer_ext)
endif ()

if (FETCH_CUTLASS)
    if (EXISTS "${LOCAL_CUTLASS_DIR}/include")
        message(STATUS "[MY_LLAMA] Found LOCAL CUTLASS Header-Only cache (${CUTLASS_TAG}) at ${LOCAL_CUTLASS_DIR}!")

        FetchContent_Declare(
                cutlass_download
                SOURCE_DIR "${LOCAL_CUTLASS_DIR}"
        )
    else ()
        message(STATUS "[MY_LLAMA] Local CUTLASS (${CUTLASS_TAG}) not found. Downloading archive...")
        FetchContent_Declare(
                cutlass_download
                URL "${NT_HOST}/NVIDIA/cutlass/archive/refs/tags/${CUTLASS_TAG}.zip"
                SOURCE_DIR "${LOCAL_CUTLASS_DIR}"
        )
    endif ()

    set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable only CUTLASS headers" FORCE)
    set(CUTLASS_ENABLE_EXAMPLES OFF CACHE BOOL "Disable CUTLASS examples" FORCE)
    set(CUTLASS_ENABLE_TESTS OFF CACHE BOOL "Disable CUTLASS tests" FORCE)

    FetchContent_MakeAvailable(cutlass_download)

    include_directories(${LOCAL_CUTLASS_DIR}/include)
    include_directories(${LOCAL_CUTLASS_DIR}/tools/util/include)
endif ()
