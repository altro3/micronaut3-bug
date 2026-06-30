#include "kernels.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

extern "C" cublasHandle_t get_global_cublas_handle();

extern "C" {
void launch_matmul_backward_weights(float *d_weights,
                                    const float *input,
                                    const float *d_output,
                                    int batch_size,
                                    int out_features,
                                    int in_features,
                                    void *stream_ptr) {
    const auto stream = static_cast<cudaStream_t>(stream_ptr);

    const cublasHandle_t handle = get_global_cublas_handle();
    cublasSetStream(handle, stream);

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    const cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        out_features,
        in_features,
        batch_size,
        &alpha,
        d_output, CUDA_R_32F, out_features,
        input, CUDA_R_32F, in_features,
        &beta,
        d_weights, CUDA_R_32F, out_features,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Ошибка cuBLAS GemmEx в backward_weights! Код: %d\n", status);
    }
}

void launch_matmul_backward_input(float *d_input,
                                  const float *d_output,
                                  const float *weights,
                                  int batch_size,
                                  int out_features,
                                  int in_features,
                                  void *stream_ptr) {
    const auto stream = static_cast<cudaStream_t>(stream_ptr);

    const cublasHandle_t handle = get_global_cublas_handle();
    cublasSetStream(handle, stream);

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    const cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        in_features,
        batch_size,
        out_features,
        &alpha,
        weights, CUDA_R_32F, out_features,
        d_output, CUDA_R_32F, out_features,
        &beta,
        d_input, CUDA_R_32F, in_features,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Ошибка cuBLAS GemmEx in backward_input! Код: %d\n", status);
    }
}
}
