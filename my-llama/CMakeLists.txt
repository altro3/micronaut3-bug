cmake_minimum_required(VERSION 4.3)
project(my_llama_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES "100;90;89")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cuda/include)

find_package(CUDAToolkit REQUIRED)

add_library(
        cuda_kernels STATIC
        cuda/include/kernels.h
        cuda/src/adamw.cu
        cuda/src/cross_entropy_loss.cu
        cuda/src/dequantize_block_gguf_q4_k.cu
        cuda/src/flash_attention_prefill_fp8.cu
        cuda/src/flash_attention_prefill_fp16.cu
        cuda/src/flash_attention_prefill_main.cu
        cuda/src/embeddings.cu
        cuda/src/flash_decoding_combine.cu
        cuda/src/flash_decoding_launcher.cu
        cuda/src/flash_decoding_partial.cu
        cuda/src/flash_decoding_write.cu
        cuda/src/fused_attention.cu
        cuda/src/fused_sampling.cu
        cuda/src/greedy_sampling.cu
        cuda/src/matmul.cu
        cuda/src/matmul_gguf_q4_k.cu
        cuda/src/prefill_kv_write.cu
        cuda/src/ragged_utils.cu
        cuda/src/rmsnorm.cu
        cuda/src/rope.cu
        cuda/src/speculative_launcher.cu
        cuda/src/speculative_verify.cu
        cuda/src/speculative_sampling.cuh
        cuda/src/swiglu.cu
)

target_link_libraries(cuda_kernels PRIVATE CUDA::cublas CUDA::cublasLt)
