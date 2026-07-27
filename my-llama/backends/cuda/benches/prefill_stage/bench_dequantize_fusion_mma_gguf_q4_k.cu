#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <iostream>
#include <ranges>

#include "data_types.h"
#include "prefill_stage/dequantize_fusion_mma_gguf_q4_k.cuh"
#include "../bench_utils.cuh"

extern "C" void launch_fused_gemm_gguf_q4_k(
    void *output_activations, const void *input_activations, const void *quantized_weights,
    int32_t batch_size_or_tokens, int32_t hidden_units_out, int32_t hidden_units_in,
    int32_t data_type, void *stream_ptr
);

static void run_benchmark(
    const int32_t data_type, const std::string &type_name,
    const int32_t M, const int32_t N, const int32_t K,
    const void *d_in, const void *d_w, void *d_out, L2CacheFlusher &flusher
) {
    constexpr int32_t warmup_iters = 10;
    constexpr int32_t bench_iters = 100;

    std::vector<float> iters_ms(bench_iters);
    GPUTimer timer;

    std::cout << "[BENCHMARK] Starting profile session for target: " << type_name << std::endl;
    std::cout << "[BENCHMARK] Matrix dimensions: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    std::cout << "[BENCHMARK] Warming up Tensor Cores (" << warmup_iters << " iterations)..." << std::endl;
    for (int32_t i = 0; i < warmup_iters; ++i) {
        launch_fused_gemm_gguf_q4_k(d_out, d_in, d_w, M, N, K, data_type, nullptr);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "[BENCHMARK] Warmup complete. Running " << bench_iters << " hot iterations..." << std::endl;

    for (int32_t i = 0; i < bench_iters; ++i) {
        flusher.flush();

        timer.start();
        launch_fused_gemm_gguf_q4_k(d_out, d_in, d_w, M, N, K, data_type, nullptr);
        timer.stop();
        iters_ms[i] = timer.elapsed_ms();
    }

    const double fops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);

    BenchmarkReporter::report_performance("GGUF Q4_K GEMM", type_name, iters_ms, fops, 0.0, true);
}

void run_q4_k_gemm_benchmarks() {
    const std::string file_path = "D:\\!models\\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf";
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

    DeviceBuffer d_w(host_real_weights);
    L2CacheFlusher flusher;

    {
        std::vector h_in(M * K, __float2bfloat16(0.1f));
        DeviceBuffer d_in(h_in);
        DeviceBuffer<__nv_bfloat16> d_out(M * N);

        run_benchmark(static_cast<int32_t>(DataType::BF16), "BF16_REAL_SCALE", M, N, K, d_in.get(), d_w.get(), d_out.get(), flusher);
    }
    {
        std::vector<uint8_t> h_in(M * K, 0x25);
        DeviceBuffer d_in(h_in);
        DeviceBuffer<__nv_bfloat16> d_out(M * N);

        run_benchmark(static_cast<int32_t>(DataType::FP8), "FP8_REAL_SCALE", M, N, K, d_in.get(), d_w.get(), d_out.get(), flusher);
    }
    {
        std::vector<uint8_t> h_in(M * K / 2, 0x11);
        DeviceBuffer d_in(h_in);
        DeviceBuffer<__nv_bfloat16> d_out(M * N);

        run_benchmark(static_cast<int32_t>(DataType::FP4), "FP4_REAL_SCALE", M, N, K, d_in.get(), d_w.get(), d_out.get(), flusher);
    }
}

REGISTER_BENCHMARK("q4_k_gemm", run_q4_k_gemm_benchmarks);
