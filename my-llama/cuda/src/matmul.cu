#include "kernels.h"
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <stdio.h>

static cublasLtHandle_t global_cublaslt_handle = nullptr;
static cublasLtMatmulDesc_t cached_operation_desc = nullptr;
static cublasLtMatrixLayout_t cached_adesc = nullptr;
static cublasLtMatrixLayout_t cached_bdesc = nullptr;
static cublasLtMatrixLayout_t cached_cdesc = nullptr;

static int cached_batch_size = 0;
static int cached_out_features = 0;
static int cached_in_features = 0;

extern "C" {
void init_cublas_infrastructure() {
    if (global_cublaslt_handle == nullptr) {
        const cublasStatus_t status = cublasLtCreate(&global_cublaslt_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "[CUBLASLT CRITICAL ERROR]: Failed to create handle! Code: %d\n", status);
        } else {
            cached_operation_desc = nullptr;
            cached_adesc = nullptr;
            cached_bdesc = nullptr;
            cached_cdesc = nullptr;
            cached_batch_size = 0;
            cached_out_features = 0;
            cached_in_features = 0;
            printf("[CUDA]: Global cuBLASLt infrastructure (Persistent Descriptor mode) successfully initialized.\n");
        }
    }
}

void destroy_cublas_infrastructure() {
    if (global_cublaslt_handle != nullptr) {
        if (cached_cdesc) cublasLtMatrixLayoutDestroy(cached_cdesc);
        if (cached_adesc) cublasLtMatrixLayoutDestroy(cached_adesc);
        if (cached_bdesc) cublasLtMatrixLayoutDestroy(cached_bdesc);
        if (cached_operation_desc) cublasLtMatmulDescDestroy(cached_operation_desc);

        cublasLtDestroy(global_cublaslt_handle);
        global_cublaslt_handle = nullptr;

        cached_operation_desc = nullptr;
        cached_adesc = nullptr;
        cached_bdesc = nullptr;
        cached_cdesc = nullptr;
        cached_batch_size = 0;
        cached_out_features = 0;
        cached_in_features = 0;
        printf("[CUDA]: cuBLASLt infrastructure successfully released.\n");
    }
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
    const cublasLtHandle_t lt_handle = global_cublaslt_handle;
    if (lt_handle == nullptr) [[unlikely]] {
        fprintf(stderr, "[CUDA ERROR]: Attempted to call launch_matmul before cuBLASLt initialization!\n");
        return;
    }

    const auto stream = static_cast<cudaStream_t>(stream_ptr);

    if (batch_size != cached_batch_size || out_features != cached_out_features || in_features != cached_in_features) [[unlikely]] {
        if (cached_cdesc) cublasLtMatrixLayoutDestroy(cached_cdesc);
        if (cached_adesc) cublasLtMatrixLayoutDestroy(cached_adesc);
        if (cached_bdesc) cublasLtMatrixLayoutDestroy(cached_bdesc);
        if (cached_operation_desc) cublasLtMatmulDescDestroy(cached_operation_desc);

        cublasLtMatmulDescCreate(&cached_operation_desc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F);
        cublasLtMatrixLayoutCreate(&cached_bdesc, CUDA_R_32F, out_features, in_features, out_features);
        cublasLtMatrixLayoutCreate(&cached_adesc, CUDA_R_32F, in_features, batch_size, in_features);
        cublasLtMatrixLayoutCreate(&cached_cdesc, CUDA_R_32F, out_features, batch_size, out_features);

        cached_batch_size = batch_size;
        cached_out_features = out_features;
        cached_in_features = in_features;
    }

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    const cublasStatus_t status = cublasLtMatmul(
        lt_handle,
        cached_operation_desc,
        &alpha,
        matrix_b, cached_bdesc,
        matrix_a, cached_adesc,
        &beta,
        output_matrix, cached_cdesc,
        output_matrix, cached_cdesc,
        nullptr,
        nullptr,
        0,
        stream
    );

    if (status != CUBLAS_STATUS_SUCCESS) [[unlikely]] {
        fprintf(stderr, "[CUDA ERROR]: Asynchronous failure in cuBLASLt matmul! Status code: %d\n", status);
    }
}
}
