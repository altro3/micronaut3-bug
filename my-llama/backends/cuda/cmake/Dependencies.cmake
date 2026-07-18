include(FetchContent)
set(NT_HOST "https://github.com")

find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED FLASHINFER_TAG)
    set(FLASHINFER_TAG "v0.6.15")
endif()

if(NOT DEFINED CUTLASS_TAG)
    set(CUTLASS_TAG "v4.6.1")
endif()

if(FETCH_FLASHINFER)
    message(STATUS "[MY_LLAMA] Fetching FlashInfer...")
    FetchContent_Declare(
            flashinfer_ext
            URL "${NT_HOST}/flashinfer-ai/flashinfer/archive/refs/tags/${FLASHINFER_TAG}.zip"
    )
    FetchContent_MakeAvailable(flashinfer_ext)
endif()

if(FETCH_CUTLASS)
    message(STATUS "[MY_LLAMA] Fetching CUTLASS...")
    FetchContent_Declare(
            cutlass
            URL "${NT_HOST}/NVIDIA/cutlass/archive/refs/tags/${CUTLASS_TAG}.zip"
            SOURCE_SUBDIR "non_existent_directory"
    )
    FetchContent_MakeAvailable(cutlass)
endif()
