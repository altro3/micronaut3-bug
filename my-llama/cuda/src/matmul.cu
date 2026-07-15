#include "kernels.h"
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <stdio.h>

enum class InferenceDtype : int {
    FP32 = 0,
    BF16 = 1,
    FP8_E4M3 = 2
};

enum class MatmulEpilogue : int {
    NONE = 0,
    GELU = 1
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
    MatmulEpilogue cached_epilogue;

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
    ctx->cached_epilogue = static_cast<MatmulEpilogue>(-1);

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
    const int epilogue_int,
    const float *a_scale_ptr,
    const float *b_scale_ptr,
    void *stream_ptr
) {
    if (!ctx) [[unlikely]] return;
    const auto stream = static_cast<cudaStream_t>(stream_ptr);
    const auto dtype = static_cast<InferenceDtype>(dtype_int);
    const auto epilogue = static_cast<MatmulEpilogue>(epilogue_int);

    if (batch_size != ctx->cached_batch_size ||
        out_features != ctx->cached_out_features ||
        in_features != ctx->cached_in_features ||
        dtype != ctx->cached_dtype ||
        epilogue != ctx->cached_epilogue) [[unlikely]] {
        if (ctx->cdesc) cublasLtMatrixLayoutDestroy(ctx->cdesc);
        if (ctx->adesc) cublasLtMatrixLayoutDestroy(ctx->adesc);
        if (ctx->bdesc) cublasLtMatrixLayoutDestroy(ctx->bdesc);
        if (ctx->operation_desc) cublasLtMatmulDescDestroy(ctx->operation_desc);

        cublasComputeType_t compute_type;
        constexpr cudaDataType_t scale_type = CUDA_R_32F;
        cudaDataType_t a_type, b_type, c_type;

        switch (dtype) {
            case InferenceDtype::FP32:
                compute_type = CUBLAS_COMPUTE_32F_FAST_16BF;
                a_type = b_type = c_type = CUDA_R_32F;
                break;
            case InferenceDtype::BF16:
                compute_type = CUBLAS_COMPUTE_32F;
                a_type = b_type = c_type = CUDA_R_16BF;
                break;
            case InferenceDtype::FP8_E4M3:
                compute_type = CUBLAS_COMPUTE_32F;
                a_type = CUDA_R_8F_E4M3;
                b_type = CUDA_R_8F_E4M3;
                c_type = CUDA_R_16BF;
                break;
            default:
                fprintf(stderr, "[CUDA ERROR]: Unsupported dtype token!\n");
                return;
        }

        cublasLtMatmulDescCreate(&ctx->operation_desc, compute_type, scale_type);

        cublasLtMatrixLayoutCreate(&ctx->bdesc, b_type, out_features, in_features, out_features);
        cublasLtMatrixLayoutCreate(&ctx->adesc, a_type, in_features, batch_size, in_features);
        cublasLtMatrixLayoutCreate(&ctx->cdesc, c_type, out_features, batch_size, out_features);

        cublasLtEpilogue_t epi_mode = CUBLASLT_EPILOGUE_DEFAULT;
        if (epilogue == MatmulEpilogue::GELU) {
            epi_mode = CUBLASLT_EPILOGUE_GELU;
        }
        cublasLtMatmulDescSetAttribute(ctx->operation_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi_mode, sizeof(epi_mode));

        ctx->cached_batch_size = batch_size;
        ctx->cached_out_features = out_features;
        ctx->cached_in_features = in_features;
        ctx->cached_dtype = dtype;
        ctx->cached_epilogue = epilogue;
    }

    if (dtype == InferenceDtype::FP8_E4M3) [[unlikely]] {
        if (a_scale_ptr) {
            cublasLtMatmulDescSetAttribute(ctx->operation_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale_ptr, sizeof(a_scale_ptr));
        }
        if (b_scale_ptr) {
            cublasLtMatmulDescSetAttribute(ctx->operation_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale_ptr, sizeof(b_scale_ptr));
        }
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
        fprintf(stderr, "[CUDA ERROR]: Universal matmul execution failed! Code: %d\n", status);
    }
}
}
