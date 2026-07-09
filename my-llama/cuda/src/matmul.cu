#include "kernels.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

static cublasHandle_t global_cublas_handle = nullptr;

extern "C" {
void init_cublas_infrastructure() {
    if (global_cublas_handle == nullptr) {
        const cublasStatus_t status = cublasCreate(&global_cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "[CUBLAS CRITICAL ERROR]: Failed to create handle! Code: %d\n", status);
        } else {
            //cublasSetMathMode(global_cublas_handle, CUBLAS_DEFAULT_MATH);
            cublasSetMathMode(global_cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);
            printf("[CUDA]: Global cuBLAS infrastructure (TF32 Tensor Cores mode) successfully initialized.\n");
        }
    }
}

void destroy_cublas_infrastructure() {
    if (global_cublas_handle != nullptr) {
        cublasDestroy(global_cublas_handle);
        global_cublas_handle = nullptr;
        printf("[CUDA]: cuBLAS infrastructure successfully released.\n");
    }
}

cublasHandle_t get_global_cublas_handle() {
    return global_cublas_handle;
}

void launch_matmul(
    float *output_matrix,
    const float *matrix_a,
    const float *matrix_b,
    const int batch_size,
    const int out_features,
    const int in_features,
    void *stream_ptr
) {
    const cublasHandle_t handle = get_global_cublas_handle();
    if (handle == nullptr) {
        fprintf(stderr, "[CUDA ERROR]: Attempted to call launch_matmul before cuBLAS initialization!\n");
        return;
    }

    const auto stream = static_cast<cudaStream_t>(stream_ptr);
    cublasSetStream(handle, stream);

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    const cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        out_features,
        batch_size,
        in_features,
        &alpha,
        matrix_b,
        CUDA_R_32F,
        out_features,
        matrix_a,
        CUDA_R_32F,
        in_features,
        &beta,
        output_matrix,
        CUDA_R_32F,
        out_features,
        //        CUBLAS_COMPUTE_32F,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[CUDA ERROR]: Asynchronous failure in cuBLAS GemmEx! Status code: %d\n", status);
    }
}
}
