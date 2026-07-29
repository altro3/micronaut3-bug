#include <doctest/doctest.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include "cutlass/bfloat16.h"
#include "cutlass/float8.h"
#include "cutlass/float_subbyte.h"
#include "prefill_stage/dequantize_fusion_mma_gguf_q4_k_blackwell.cuh"

using ElementSFA = cutlass::float_ue4m3_t;
using ElementSFB = cutlass::float_ue4m3_t;
using ElementD = cutlass::bfloat16_t;

TEST_CASE("BlackwellNativeFp4GemmTest - Verification") {
    constexpr int32_t M = 128;
    constexpr int32_t N = 256;
    constexpr int32_t K = 128;

    std::vector<uint8_t> h_A(M * K / 2, 0);
    std::vector<uint8_t> h_B(N * K / 2, 0);

    std::vector h_SFA(M * K / 32, ElementSFA(1.0f));
    std::vector h_SFB(N * K / 32, ElementSFB(1.0f));
    std::vector h_D(M * N, ElementD(0.0f));

    std::ranges::fill(h_A, 0x77);
    std::ranges::fill(h_B, 0x77);

    cudaStream_t test_stream;
    REQUIRE(cudaStreamCreate(&test_stream) == cudaSuccess);

    auto align_to_cache_line = [](const size_t size) {
        return (size + 127) / 128 * 128;
    };

    size_t size_A = align_to_cache_line(h_A.size());
    size_t size_B = align_to_cache_line(h_B.size());
    size_t size_SFA = align_to_cache_line(h_SFA.size() * sizeof(ElementSFA));
    size_t size_SFB = align_to_cache_line(h_SFB.size() * sizeof(ElementSFB));
    size_t size_D = align_to_cache_line(h_D.size() * sizeof(ElementD));

    void *d_A = nullptr;
    void *d_B = nullptr;
    void *d_SFA = nullptr;
    void *d_SFB = nullptr;
    void *d_D = nullptr;

    REQUIRE(cudaMallocAsync(&d_A, size_A, test_stream) == cudaSuccess);
    REQUIRE(cudaMallocAsync(&d_B, size_B, test_stream) == cudaSuccess);
    REQUIRE(cudaMallocAsync(&d_SFA, size_SFA, test_stream) == cudaSuccess);
    REQUIRE(cudaMallocAsync(&d_SFB, size_SFB, test_stream) == cudaSuccess);
    REQUIRE(cudaMallocAsync(&d_D, size_D, test_stream) == cudaSuccess);

    REQUIRE(cudaMemcpyAsync(d_A, h_A.data(), h_A.size(), cudaMemcpyHostToDevice, test_stream) == cudaSuccess);
    REQUIRE(cudaMemcpyAsync(d_B, h_B.data(), h_B.size(), cudaMemcpyHostToDevice, test_stream) == cudaSuccess);
    REQUIRE(cudaMemcpyAsync(d_SFA, h_SFA.data(), h_SFA.size() * sizeof(ElementSFA), cudaMemcpyHostToDevice, test_stream) == cudaSuccess);
    REQUIRE(cudaMemcpyAsync(d_SFB, h_SFB.data(), h_SFB.size() * sizeof(ElementSFB), cudaMemcpyHostToDevice, test_stream) == cudaSuccess);
    REQUIRE(cudaMemsetAsync(d_D, 0, size_D, test_stream) == cudaSuccess);

    launch_blackwell_fp4_native_gemm(
        d_D,
        d_A,
        d_B,
        d_SFA,
        d_SFB,
        M, N, K,
        test_stream
    );

    REQUIRE(cudaStreamSynchronize(test_stream) == cudaSuccess);
    REQUIRE(cudaMemcpyAsync(h_D.data(), d_D, h_D.size() * sizeof(ElementD), cudaMemcpyDeviceToHost, test_stream) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(test_stream) == cudaSuccess);

    float sample_actual = h_D[0];
    std::cout << "[ENGINE INFO] Blackwell Hardware MMA Output: " << sample_actual << std::endl;

    bool has_error = false;
    if (sample_actual == 0.0f) {
        std::cerr << "[ERROR] Kernel executed but returned absolute zeros. Tensor Cores are bypassed!" << std::endl;
        has_error = true;
    }

    CHECK_FALSE(has_error);

    cudaFreeAsync(d_A, test_stream);
    cudaFreeAsync(d_B, test_stream);
    cudaFreeAsync(d_SFA, test_stream);
    cudaFreeAsync(d_SFB, test_stream);
    cudaFreeAsync(d_D, test_stream);
    cudaStreamDestroy(test_stream);
}
