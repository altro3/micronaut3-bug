#include <doctest/doctest.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <iostream>
#include <vector>
#include "prefill_stage/mini_tensor_test.cuh"

TEST_CASE("MiniTensorCoreVerificationTest - Split Compilation") {
    std::cout << "\n=========================================" << std::endl;
    std::cout << "=== [STARTING TENSOR CORE MINI BENCH] ===" << std::endl;
    std::cout << "=========================================" << std::endl;

    constexpr int M = 16;
    constexpr int N = 16;
    constexpr int K = 16;

    std::vector<__nv_bfloat16> h_A(M * K);
    std::vector<__nv_bfloat16> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);

    for (int i = 0; i < M * K; ++i) h_A[i] = __float2bfloat16(1.0f);
    for (int i = 0; i < K * N; ++i) h_B[i] = __float2bfloat16(2.0f);

    REQUIRE(cudaSetDevice(0) == cudaSuccess);

    cudaFree(0);
    cudaDeviceSynchronize();

    float *d_C = nullptr;
    void *d_A = nullptr;
    void *d_B = nullptr;

    REQUIRE(cudaMalloc(&d_A, h_A.size() * sizeof(__nv_bfloat16)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_B, h_B.size() * sizeof(__nv_bfloat16)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_C, h_C.size() * sizeof(float)) == cudaSuccess);

    REQUIRE(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemset(d_C, 0, h_C.size() * sizeof(float)) == cudaSuccess);

    cudaError_t launch_err = launch_mini_wmma_bf16(d_C, d_A, d_B);
    std::cout << "[CUDA] Call finished with code: " << launch_err << std::endl;
    REQUIRE(launch_err == cudaSuccess);

    cudaError_t sync_err = cudaDeviceSynchronize();
    std::cout << "[CUDA] Device sync status: " << sync_err << std::endl;
    REQUIRE(sync_err == cudaSuccess);

    REQUIRE(cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);

    std::cout << "\n=== [TENSOR CORE HARDWARE MATRIX OUT] ===" << std::endl;
    std::cout << "Expected cell value (1.0 * 2.0 * 16): 32" << std::endl;
    std::cout << "Actual hardware value at index 0:     " << h_C[0] << std::endl;
    std::cout << "=========================================\n" << std::endl;

    REQUIRE(h_C[0] == 32.0f);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
