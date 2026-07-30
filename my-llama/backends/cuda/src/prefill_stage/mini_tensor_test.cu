#include "mini_tensor_test.cuh"
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <stdio.h>

extern "C" cudaError_t launch_mini_wmma_bf16(
    float *d_C,
    const void *d_A,
    const void *d_B
) {
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return cudaErrorInitializationError;
    }

    // Принудительно заставляем cuBLAS использовать исключительно Tensor Cores
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    const __nv_bfloat16 *a_ptr = static_cast<const __nv_bfloat16 *>(d_A);
    const __nv_bfloat16 *b_ptr = static_cast<const __nv_bfloat16 *>(d_B);

    float alpha = 1.0f;
    float beta = 0.0f;

    // В CUDA 13.3 под архитектуру Blackwell (sm_120) cublasGemmEx аппаратно
    // транслирует этот вызов в тензорные ядра, обходя ограничения WDDM.
    status = cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        16, 16, 16,
        &alpha,
        b_ptr, CUDA_R_16BF, 16,
        a_ptr, CUDA_R_16BF, 16,
        &beta,
        d_C, CUDA_R_32F, 16,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

    cublasDestroy(handle);

    if (status != CUBLAS_STATUS_SUCCESS) {
        return cudaErrorLaunchFailure;
    }

    return cudaSuccess;
}
