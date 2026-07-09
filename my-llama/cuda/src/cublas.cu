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
            fprintf(stderr, "[КРИТИЧЕСКАЯ ОШИБКА]: Не удалось инициализировать cuBLAS контекст! Код: %d\n", status);
        } else {
            printf("[MY-LLAMA] Аппаратный контекст cuBLAS для Tensor Cores успешно создан.\n");
        }
    }
}


cublasHandle_t get_global_cublas_handle() {
    if (global_cublas_handle == nullptr) {
        init_cublas_infrastructure();
    }
    return global_cublas_handle;
}
}
