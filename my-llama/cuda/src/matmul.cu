#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

static cublasHandle_t global_cublas_handle = nullptr;

extern "C" void init_cublas_infrastructure() {
    if (global_cublas_handle == nullptr) {
        const cublasStatus_t status = cublasCreate(&global_cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "[КРИТИЧЕСКАЯ ОШИБКА]: Не удалось инициализировать cuBLAS контекст! Код: %d\n", status);
        } else {
            printf("[MY-LLAMA] Аппаратный контекст cuBLAS для Tensor Cores успешно создан.\n");
        }
    }
}

__global__ void update_kv_cache_kernel(float *const k_cache, float *const v_cache, const float *const new_k, const float *const new_v, const int token_index, const int hidden_size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_size) {
        const int cache_offset = token_index * hidden_size + idx;
        k_cache[cache_offset] = new_k[idx];
        v_cache[cache_offset] = new_v[idx];
    }
}

__global__ void residual_kernel(float *const input_output, const float *const residual_data, const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) { input_output[idx] += residual_data[idx]; }
}

extern "C" {
void launch_matmul(float *output_matrix,
                   const float *matrix_a,
                   const float *matrix_b,
                   const int batch_size, const int out_features, const int in_features,
                   void *stream_ptr) {
    if (global_cublas_handle == nullptr) {
        init_cublas_infrastructure();
    }

    const auto stream = static_cast<cudaStream_t>(stream_ptr);

    cublasSetStream(global_cublas_handle, stream);

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    const cublasStatus_t status = cublasGemmEx(
        global_cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        out_features,
        batch_size,
        in_features,
        &alpha,
        matrix_b, CUDA_R_32F, out_features,
        matrix_a, CUDA_R_32F, in_features,
        &beta,
        output_matrix, CUDA_R_32F, out_features,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Ошибка асинхронного cuBLAS GemmEx! Код статуса: %d\n", status);
    }
}

void launch_update_kv_cache(float *k_cache, float *v_cache, const float *new_k, const float *new_v,
                            const int token_index, const int hidden_size, void *stream_ptr) {
    constexpr int threads = 256;
    const int blocks = (hidden_size + threads - 1) / threads;
    auto stream = static_cast<cudaStream_t>(stream_ptr);

    update_kv_cache_kernel<<<blocks, threads, 0, stream>>>(k_cache, v_cache, new_k, new_v, token_index, hidden_size);
}

void launch_residual(float *input_output, const float *residual_data, const int size, void *stream_ptr) {
    constexpr int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    auto stream = static_cast<cudaStream_t>(stream_ptr);

    residual_kernel<<<blocks, threads, 0, stream>>>(input_output, residual_data, size);
}
}
