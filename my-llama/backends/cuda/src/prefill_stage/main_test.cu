#include "mini_tensor_test.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <vector>

int main() {
    if (cudaSetDevice(0) != cudaSuccess) {
        printf("[HOST ERROR] Failed to set device 0\n");
        return -1;
    }

    cudaFree(0);

    constexpr int M = 16, N = 16, K = 16;
    std::vector<__nv_bfloat16> h_A(M * K);
    std::vector<__nv_bfloat16> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);

    for (int i = 0; i < M * K; ++i) h_A[i] = __float2bfloat16(1.0f);
    for (int i = 0; i < K * N; ++i) h_B[i] = __float2bfloat16(2.0f);

    float *d_C = nullptr;
    void *d_A = nullptr, *d_B = nullptr;

    cudaMalloc(&d_A, h_A.size() * sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, h_B.size() * sizeof(__nv_bfloat16));
    cudaMalloc(&d_C, h_C.size() * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, h_C.size() * sizeof(float));

    printf("[HOST] Launching C++ CuTe GEMM kernel on Tensor Cores via Interface...\n");

    launch_cute_blackwell_gemm(d_C, d_A, d_B);

    cudaError_t launch_err = cudaGetLastError();
    printf("[HOST] Launch status: %d (%s)\n", launch_err, cudaGetErrorString(launch_err));

    cudaError_t sync_err = cudaDeviceSynchronize();
    printf("[HOST] Sync status: %d (%s)\n", sync_err, cudaGetErrorString(sync_err));

    cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\n=== [VERIFICATION] ===\n");
    printf("Expected cell value: 32.000000\n");
    printf("Actual cell value:   %f\n", h_C[0]);
    printf("=========================================\n\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
