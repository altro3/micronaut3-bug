#include <doctest/doctest.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <vector>
#include <fstream>
#include <string>
#include "data_types.h"
#include "prefill_stage/dequantize_fusion_mma_gguf_q4_k.cuh"
#include <algorithm>
#include <iostream>
#include <ranges>

extern "C" void launch_fused_gemm_gguf_q4_k(
    void *output_activations, const void *input_activations, const void *quantized_weights,
    int32_t batch_size_or_tokens, int32_t hidden_units_out, int32_t hidden_units_in,
    int32_t data_type, void *stream_ptr
);

static void run_benchmark(const int32_t data_type, const std::string &type_name, const int32_t M, const int32_t N, const int32_t K, const void *d_in, const void *d_w, void *d_out) {
    constexpr int32_t warmup_iters = 10;
    constexpr int32_t bench_iters = 100;
    std::vector<float> iters_ms(bench_iters);

    std::cout << "[BENCHMARK] Starting profile session for target: " << type_name << std::endl;
    std::cout << "[BENCHMARK] Matrix dimensions: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    std::cout << "[BENCHMARK] Warming up Tensor Cores (" << warmup_iters << " iterations)..." << std::endl;
    for (int32_t i = 0; i < warmup_iters; ++i) {
        launch_fused_gemm_gguf_q4_k(d_out, d_in, d_w, M, N, K, data_type, nullptr);
    }
    cudaDeviceSynchronize();

    std::cout << "[BENCHMARK] Warmup complete. Running " << bench_iters << " hot iterations..." << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int32_t i = 0; i < bench_iters; ++i) {
        cudaEventRecord(start, nullptr);
        launch_fused_gemm_gguf_q4_k(d_out, d_in, d_w, M, N, K, data_type, nullptr);
        cudaEventRecord(stop, nullptr);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        iters_ms[i] = ms;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaDeviceSynchronize();

    std::ranges::sort(iters_ms);

    float sum_time = 0.0f;
    for (const float t: iters_ms) sum_time += t;
    const float avg_time_ms = sum_time / bench_iters;

    const float p50 = iters_ms[static_cast<int32_t>(bench_iters * 0.50)];
    const float p90 = iters_ms[static_cast<int32_t>(bench_iters * 0.90)];
    const float p95 = iters_ms[static_cast<int32_t>(bench_iters * 0.95)];

    const double fops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    const double avg_gflops = fops * 1e-9 / (static_cast<double>(avg_time_ms) * 1e-3);

    std::cout << "==========================================================================" << std::endl;
    std::cout << "[BENCHMARK RESULTS - " << type_name << "]" << std::endl;
    std::cout << "  Average Time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "  Percentile 50% (Median): " << p50 << " ms" << std::endl;
    std::cout << "  Percentile 90%: " << p90 << " ms" << std::endl;
    std::cout << "  Percentile 95%: " << p95 << " ms" << std::endl;
    std::cout << "  Average Compute Performance: " << avg_gflops << " GFLOPs" << std::endl;
    std::cout << "==========================================================================" << std::endl;
}

TEST_CASE("GgufBenchmarkTest - RealLLamStyleBench") {
    const std::string file_path = "D:\\!models\\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"; // Или путь к Llama 8B
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "[SKIP Benchmark] File not found at " << file_path << std::endl;
        return;
    }

    constexpr int32_t M = 1024;
    constexpr int32_t N = 4096;
    constexpr int32_t K = 4096;

    constexpr size_t total_weight_elements = N * K;
    constexpr size_t total_blocks = total_weight_elements / 256;
    std::vector<BlockQ4K> host_real_weights(total_blocks);

    file.seekg(1024 * 1024 * 10);
    file.read(reinterpret_cast<char *>(host_real_weights.data()), total_blocks * sizeof(BlockQ4K));
    file.close();

    void *d_w = nullptr;
    REQUIRE(cudaMalloc(&d_w, host_real_weights.size() * sizeof(BlockQ4K)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_w, host_real_weights.data(), host_real_weights.size() * sizeof(BlockQ4K), cudaMemcpyHostToDevice) == cudaSuccess);

    {
        std::vector<__nv_bfloat16> h_in(M * K, __float2bfloat16(0.1f));
        void *d_in = nullptr;
        void *d_out = nullptr;
        REQUIRE(cudaMalloc(&d_in, M * K * sizeof(__nv_bfloat16)) == cudaSuccess);
        REQUIRE(cudaMalloc(&d_out, M * N * sizeof(__nv_bfloat16)) == cudaSuccess);
        REQUIRE(cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice) == cudaSuccess);

        run_benchmark(static_cast<int32_t>(DataType::BF16), "BF16_REAL_SCALE", M, N, K, d_in, d_w, d_out);

        cudaFree(d_in);
        cudaFree(d_out);
    }
    {
        std::vector<uint8_t> h_in(M * K, 0x25);
        void *d_in = nullptr;
        void *d_out = nullptr;
        REQUIRE(cudaMalloc(&d_in, M * K * sizeof(uint8_t)) == cudaSuccess);
        REQUIRE(cudaMalloc(&d_out, M * N * sizeof(__nv_bfloat16)) == cudaSuccess);
        REQUIRE(cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(uint8_t), cudaMemcpyHostToDevice) == cudaSuccess);

        run_benchmark(static_cast<int32_t>(DataType::FP8), "FP8_REAL_SCALE", M, N, K, d_in, d_w, d_out);

        cudaFree(d_in);
        cudaFree(d_out);
    }
    {
        std::vector<uint8_t> h_in(M * K / 2, 0x11);
        void *d_in = nullptr;
        void *d_out = nullptr;
        REQUIRE(cudaMalloc(&d_in, M * K / 2 * sizeof(uint8_t)) == cudaSuccess);
        REQUIRE(cudaMalloc(&d_out, M * N * sizeof(__nv_bfloat16)) == cudaSuccess);
        REQUIRE(cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(uint8_t), cudaMemcpyHostToDevice) == cudaSuccess);

        run_benchmark(static_cast<int32_t>(DataType::FP4), "FP4_REAL_SCALE", M, N, K, d_in, d_w, d_out);

        cudaFree(d_in);
        cudaFree(d_out);
    }

    cudaFree(d_w);
}
