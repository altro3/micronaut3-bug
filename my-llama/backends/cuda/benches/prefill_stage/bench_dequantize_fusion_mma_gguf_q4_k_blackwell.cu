#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <string>
#include <iostream>
#include "../bench_utils.cuh"
#include "prefill_stage/dequantize_fusion_mma_gguf_q4_k_blackwell.cuh"

void run_gguf_blackwell_native_benchmarks() {
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

    const DeviceBuffer d_w(host_real_weights);
    const DeviceBuffer<uint16_t> d_in(M * K, 0x3C00);
    const DeviceBuffer<uint16_t> d_out(M * N, 0);

    constexpr int32_t warmup_iters = 10;
    constexpr int32_t bench_iters = 100;
    std::vector<float> iters_ms(bench_iters);
    const GPUTimer timer;
    const L2CacheFlusher flusher;

    std::cout << "[BENCHMARK] Profiling GGUF to Blackwell Native Block-Scaled Pipeline" << std::endl;

    for (int32_t i = 0; i < warmup_iters; ++i) {
        launch_fused_gemm_gguf_blackwell_fp4_native(
            d_out.get_void(),
            d_in.get_const_void(),
            d_w.get_const_void(),
            M, N, K,
            nullptr
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int32_t i = 0; i < bench_iters; ++i) {
        flusher.flush();

        timer.start();
        launch_fused_gemm_gguf_blackwell_fp4_native(
            d_out.get_void(),
            d_in.get_const_void(),
            d_w.get_const_void(),
            M, N, K,
            nullptr
        );
        timer.stop();
        iters_ms[i] = timer.elapsed_ms();
    }

    constexpr double fops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    BenchmarkReporter::report_performance("BLACKWELL GGUF STAGED", "SM120_FP4", iters_ms, fops, 0.0, false);
}

REGISTER_BENCHMARK("gguf_blackwell", run_gguf_blackwell_native_benchmarks);
