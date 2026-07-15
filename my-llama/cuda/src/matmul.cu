#include "kernels.h"
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <stdio.h>

enum class InferenceDtype : int {
    FP32 = 0,
    BF16 = 1,
    FP8_E4M3 = 2
};

struct MatmulContext {
    cublasLtHandle_t lt_handle;
    cublasLtMatmulDesc_t operation_desc;
    cublasLtMatrixLayout_t adesc;
    cublasLtMatrixLayout_t bdesc;
    cublasLtMatrixLayout_t cdesc;

    int cached_batch_size;
    int cached_out_features;
    int cached_in_features;
    InferenceDtype cached_dtype;

    void *workspace;
    size_t workspace_size;
};

extern "C" {
MatmulContext *create_matmul_context(const size_t workspace_size) {
    const auto ctx = new MatmulContext();
    if (cublasLtCreate(&ctx->lt_handle) != CUBLAS_STATUS_SUCCESS) {
        delete ctx;
        return nullptr;
    }
    ctx->operation_desc = nullptr;
    ctx->adesc = nullptr;
    ctx->bdesc = nullptr;
    ctx->cdesc = nullptr;
    ctx->cached_batch_size = 0;
    ctx->cached_out_features = 0;
    ctx->cached_in_features = 0;
    ctx->cached_dtype = static_cast<InferenceDtype>(-1);

    ctx->workspace_size = workspace_size;
    ctx->workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&ctx->workspace, workspace_size);
    }
    return ctx;
}

void destroy_matmul_context(MatmulContext *ctx) {
    if (!ctx) return;
    if (ctx->cdesc) cublasLtMatrixLayoutDestroy(ctx->cdesc);
    if (ctx->adesc) cublasLtMatrixLayoutDestroy(ctx->adesc);
    if (ctx->bdesc) cublasLtMatrixLayoutDestroy(ctx->bdesc);
    if (ctx->operation_desc) cublasLtMatmulDescDestroy(ctx->operation_desc);
    if (ctx->workspace) cudaFree(ctx->workspace);
    cublasLtDestroy(ctx->lt_handle);
    delete ctx;
}

void launch_matmul_universal(
    MatmulContext *ctx,
    void *output_matrix,
    const void *matrix_a,
    const void *matrix_b,
    const int batch_size,
    const int out_features,
    const int in_features,
    const int dtype_int,
    void *stream_ptr
) {
    if (!ctx) [[unlikely]] return;
    const auto stream = static_cast<cudaStream_t>(stream_ptr);
    const auto dtype = static_cast<InferenceDtype>(dtype_int);

    if (batch_size != ctx->cached_batch_size ||
        out_features != ctx->cached_out_features ||
        in_features != ctx->cached_in_features ||
        dtype != ctx->cached_dtype) [[unlikely]] {
        if (ctx->cdesc) cublasLtMatrixLayoutDestroy(ctx->cdesc);
        if (ctx->adesc) cublasLtMatrixLayoutDestroy(ctx->adesc);
        if (ctx->bdesc) cublasLtMatrixLayoutDestroy(ctx->bdesc);
        if (ctx->operation_desc) cublasLtMatmulDescDestroy(ctx->operation_desc);

        cublasLtMatmulDescCreate(&ctx->operation_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

        cudaDataType_t a_type, b_type, c_type;

        switch (dtype) {
            case InferenceDtype::FP32:
                a_type = b_type = c_type = CUDA_R_32F;
                break;
            case InferenceDtype::BF16:
                a_type = b_type = c_type = CUDA_R_16BF;
                break;
            case InferenceDtype::FP8_E4M3:
                a_type = CUDA_R_8F_E4M3;
                b_type = CUDA_R_8F_E4M3;
                c_type = CUDA_R_16BF;
                break;
            default:
                fprintf(stderr, "[CUDA ERROR]: Unsupported dtype token!\n");
                return;
        }

        cublasLtMatrixLayoutCreate(&ctx->bdesc, b_type, out_features, in_features, out_features);
        cublasLtMatrixLayoutCreate(&ctx->adesc, a_type, in_features, batch_size, in_features);
        cublasLtMatrixLayoutCreate(&ctx->cdesc, c_type, out_features, batch_size, out_features);

        ctx->cached_batch_size = batch_size;
        ctx->cached_out_features = out_features;
        ctx->cached_in_features = in_features;
        ctx->cached_dtype = dtype;
    }

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    const cublasStatus_t status = cublasLtMatmul(
        ctx->lt_handle,
        ctx->operation_desc,
        &alpha,
        matrix_b, ctx->bdesc,
        matrix_a, ctx->adesc,
        &beta,
        output_matrix, ctx->cdesc,
        output_matrix, ctx->cdesc,
        nullptr,
        ctx->workspace,
        ctx->workspace_size,
        stream
    );

    if (status != CUBLAS_STATUS_SUCCESS) [[unlikely]] {
        fprintf(stderr, "[CUDA ERROR]: Universal matmul failed! Code: %d\n", status);
    }
}
}
