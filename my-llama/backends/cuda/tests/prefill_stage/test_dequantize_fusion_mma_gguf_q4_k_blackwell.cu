#include <doctest.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include "cutlass/bfloat16.h"
#include "cutlass/float8.h"

extern "C" void launch_blackwell_fp4_native_gemm(
    void *output_d,
    const void *input_a,
    const void *weights_b,
    const void *scales_a,
    const void *scales_b,
    int32_t m_extent,
    int32_t n_extent,
    int32_t k_extent,
    void *stream_ptr
);

using ElementSFA = cutlass::float_ue4m3_t;
using ElementSFB = cutlass::float_ue4m3_t;
using ElementD = cutlass::bfloat16_t;

TEST_CASE("BlackwellNativeFp4GemmTest - Verification") {
    int device = 0;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
        int clock_khz = 0;
        int mem_clock_khz = 0;
        cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device);
        cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, device);

        std::cout << "\n========================================================" << std::endl;
        std::cout << "=== [HARDWARE IN-DEPTH ARCHITECTURAL INSPECTION] ===" << std::endl;
        std::cout << "========================================================" << std::endl;
        std::cout << "[GPU CORE]" << std::endl;
        std::cout << "  Name:                               " << prop.name << std::endl;
        std::cout << "  Compute Capability:                 " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  SM (Multiprocessor) Count:          " << prop.multiProcessorCount << std::endl;
        std::cout << "  Warp Size:                          " << prop.warpSize << " threads" << std::endl;
        std::cout << "  Max Threads per SM:                 " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  Max Threads per Block:              " << prop.maxThreadsPerBlock << std::endl;

        std::cout << "\n[MEMORY SUB-SYSTEM]" << std::endl;
        std::cout << "  Total Global Memory:                " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  L2 Cache Size:                      " << prop.l2CacheSize / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Memory Bus Width:                   " << prop.memoryBusWidth << " bit" << std::endl;
        std::cout << "  Core Clock Rate:                    " << clock_khz / 1000 << " MHz" << std::endl;
        std::cout << "  Memory Clock Rate:                  " << mem_clock_khz / 1000 << " MHz" << std::endl;

        std::cout << "\n[SHARED MEMORY & REGISTERS]" << std::endl;
        std::cout << "  Max Shared Memory per Block (Static): " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Max Shared Memory per SM (Dynamic):   " << prop.sharedMemPerMultiprocessor / 1024 << " KB" << std::endl;
        std::cout << "  Max Shared Memory per Block (OptIn):  " << prop.sharedMemPerBlockOptin / 1024 << " KB" << std::endl;
        std::cout << "  Max Registers per Block (32-bit):   " << prop.regsPerBlock << std::endl;
        std::cout << "  Max Registers per SM (32-bit):      " << prop.regsPerMultiprocessor << std::endl;

        std::cout << "\n[BLACKWELL/HOPPER HARDWARE GRID FEATS]" << std::endl;
        std::cout << "  Max Thread Block Cluster Size X:    " << prop.maxGridSize[0] << std::endl;
        std::cout << "  Cooperative Launch Support:         " << (prop.cooperativeLaunch ? "ENABLED" : "DISABLED") << std::endl;
        std::cout << "  Unified Addressing (UVA):           " << (prop.unifiedAddressing ? "ENABLED" : "DISABLED") << std::endl;
        std::cout << "  ECC Protection Status:              " << (prop.ECCEnabled ? "ENABLED" : "DISABLED") << std::endl;
        std::cout << "========================================================\n" << std::endl;
    } else {
        std::cerr << "[HARDWARE ERROR] Failed to fetch device properties!" << std::endl;
    }

    constexpr int32_t M = 1024;
    constexpr int32_t N = 4096;
    constexpr int32_t K = 4096;

    std::vector<uint8_t> h_A(M * K / 2, 0);
    std::vector<uint8_t> h_B(N * K / 2, 0);

    std::vector<ElementSFA> h_SFA(M * K / 16, ElementSFA(1.0f));
    std::vector<ElementSFB> h_SFB(N * K / 16, ElementSFB(1.0f));
    std::vector<ElementD> h_D(M * N, ElementD(0.0f));

    std::ranges::fill(h_A, 0x33);
    std::ranges::fill(h_B, 0x33);

    std::ranges::fill(h_SFA, ElementSFA(1.0f));
    std::ranges::fill(h_SFB, ElementSFB(1.0f));

    cudaStream_t test_stream;
    REQUIRE(cudaStreamCreate(&test_stream) == cudaSuccess);

    void *d_A = nullptr;
    void *d_B = nullptr;
    void *d_SFA = nullptr;
    void *d_SFB = nullptr;
    void *d_D = nullptr;

    REQUIRE(cudaMalloc(&d_A, h_A.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_B, h_B.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_SFA, h_SFA.size() * sizeof(ElementSFA)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_SFB, h_SFB.size() * sizeof(ElementSFB)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_D, h_D.size() * sizeof(ElementD)) == cudaSuccess);

    REQUIRE(cudaMemcpyAsync(d_A, h_A.data(), h_A.size(), cudaMemcpyHostToDevice, test_stream) == cudaSuccess);
    REQUIRE(cudaMemcpyAsync(d_B, h_B.data(), h_B.size(), cudaMemcpyHostToDevice, test_stream) == cudaSuccess);
    REQUIRE(cudaMemcpyAsync(d_SFA, h_SFA.data(), h_SFA.size() * sizeof(ElementSFA), cudaMemcpyHostToDevice, test_stream) == cudaSuccess);
    REQUIRE(cudaMemcpyAsync(d_SFB, h_SFB.data(), h_SFB.size() * sizeof(ElementSFB), cudaMemcpyHostToDevice, test_stream) == cudaSuccess);
    REQUIRE(cudaMemsetAsync(d_D, 0, h_D.size() * sizeof(ElementD), test_stream) == cudaSuccess);

    launch_blackwell_fp4_native_gemm(
        d_D,
        d_A,
        d_B,
        d_SFA,
        d_SFB,
        M, N, K,
        test_stream
    );

    cudaError_t kernel_err = cudaGetLastError();
    REQUIRE(kernel_err == cudaSuccess);

    REQUIRE(cudaStreamSynchronize(test_stream) == cudaSuccess);
    REQUIRE(cudaMemcpyAsync(h_D.data(), d_D, h_D.size() * sizeof(ElementD), cudaMemcpyDeviceToHost, test_stream) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(test_stream) == cudaSuccess);

    float sample_actual = h_D[0];
    std::cout << "[ENGINE INFO] Blackwell Hardware MMA Output: " << sample_actual << std::endl;

    bool has_error = false;
    if (sample_actual == 0.0f) {
        std::cerr << "[ERROR] Kernel executed but returned absolute zeros!" << std::endl;
        has_error = true;
    }

    CHECK_FALSE(has_error);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_SFA);
    cudaFree(d_SFB);
    cudaFree(d_D);
    cudaStreamDestroy(test_stream);
}
