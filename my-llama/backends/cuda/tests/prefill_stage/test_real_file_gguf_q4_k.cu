#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <vector>
#include <fstream>
#include <string>
#include "data_types.h"
#include "prefill_stage/dequantize_fusion_mma_gguf_q4_k.cuh"

extern "C" void launch_fused_gemm_gguf_q4_k(
    void *output_activations, const void *input_activations, const void *quantized_weights,
    int32_t batch_size_or_tokens, int32_t hidden_units_out, int32_t hidden_units_in,
    int32_t data_type, void *stream_ptr
);

static void run_benchmark(const int32_t data_type, const std::string &type_name, const int32_t M, const int32_t N, const int32_t K, const void *d_in, const void *d_w, void *d_out) {
    constexpr int32_t warmup_iters = 10;
    constexpr int32_t bench_iters = 100;

    for (int32_t i = 0; i < warmup_iters; ++i) {
        launch_fused_gemm_gguf_q4_k(d_out, d_in, d_w, M, N, K, data_type, nullptr);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, nullptr);
    for (int32_t i = 0; i < bench_iters; ++i) {
        launch_fused_gemm_gguf_q4_k(d_out, d_in, d_w, M, N, K, data_type, nullptr);
        cudaStreamSynchronize(nullptr);
    }
    cudaEventRecord(stop, nullptr);
    cudaDeviceSynchronize();

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    const float avg_time_ms = milliseconds / bench_iters;

    const double fops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    const double gflops = fops * 1e-9 / (static_cast<double>(avg_time_ms) * 1e-3);

    std::cout << "[BENCHMARK " << type_name << "] Avg Time: " << avg_time_ms << " ms | Performance: " << gflops << " GFLOPs" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

TEST(GgufBenchmarkTest, BenchQwen05B_AllTypes) {
    const std::string file_path = "D:\\!models\\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf";
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "[SKIP Benchmark] File not found at " << file_path << std::endl;
        SUCCEED();
        return;
    }

    constexpr int32_t M = 1;
    constexpr int32_t N = 896;
    constexpr int32_t K = 896;

    constexpr size_t total_weight_elements = N * K;
    constexpr size_t total_blocks = total_weight_elements / 256;
    std::vector<BlockQ4K> host_real_weights(total_blocks);

    file.seekg(1024 * 1024 * 2);
    file.read(reinterpret_cast<char *>(host_real_weights.data()), total_blocks * sizeof(BlockQ4K));
    file.close();

    void *d_w;
    ASSERT_EQ(cudaMalloc(&d_w, host_real_weights.size() * sizeof(BlockQ4K)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_w, host_real_weights.data(), host_real_weights.size() * sizeof(BlockQ4K), cudaMemcpyHostToDevice), cudaSuccess);

    // {
    //     std::vector h_in(M * K, __float2bfloat16(0.01f));
    //     void *d_in, *d_out;
    //     cudaMalloc(&d_in, M * K * sizeof(__nv_bfloat16));
    //     cudaMalloc(&d_out, M * N * sizeof(__nv_bfloat16));
    //     cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    //
    //     run_benchmark(static_cast<int32_t>(DataType::BF16), "BF16", M, N, K, d_in, d_w, d_out);
    //
    //     cudaFree(d_in);
    //     cudaFree(d_out);
    // }
    //
    // {
    //     std::vector h_in(M * K, static_cast<__nv_fp8_e4m3>(0.25f));
    //     void *d_in, *d_out;
    //     cudaMalloc(&d_in, M * K * sizeof(__nv_fp8_e4m3));
    //     cudaMalloc(&d_out, M * N * sizeof(__nv_fp8_e4m3));
    //     cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
    //
    //     run_benchmark(static_cast<int32_t>(DataType::FP8), "FP8", M, N, K, d_in, d_w, d_out);
    //
    //     cudaFree(d_in);
    //     cudaFree(d_out);
    // }

    {
        const std::vector<uint8_t> h_in(M * K / 2, 0x11);
        void *d_in, *d_out;
        cudaMalloc(&d_in, M * K / 2 * sizeof(uint8_t));
        cudaMalloc(&d_out, M * N / 2 * sizeof(uint8_t));
        cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);

        run_benchmark(static_cast<int32_t>(DataType::FP4), "FP4", M, N, K, d_in, d_w, d_out);

        cudaFree(d_in);
        cudaFree(d_out);
    }

    cudaFree(d_w);
}
