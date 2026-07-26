#include <doctest/doctest.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <ranges>
#include "data_types.h"

extern "C" {
void launch_fused_rmsnorm_forward(
    void *out, const void *input, const void *gamma, const float *gamma_scales,
    float epsilon, int32_t total_tokens, int32_t hidden_size,
    int32_t data_type, int32_t threads_per_block, void *stream_ptr
);
}

static void run_rmsnorm_benchmark(
    const int32_t data_type, const std::string &type_name,
    const int32_t total_tokens, const int32_t hidden_size,
    const void *d_in_base, const void *d_gamma, const float *d_gamma_scales,
    void *d_out_base, size_t single_input_bytes
) {
    constexpr int32_t warmup_iters = 50;
    constexpr int32_t bench_iters = 200;
    constexpr int32_t threads_per_block = 64;
    constexpr float epsilon = 1e-5f;
    std::vector<float> iters_ms(bench_iters);

    size_t single_output_bytes = total_tokens * hidden_size * sizeof(__nv_bfloat16);
    constexpr int32_t num_buffers = 8;

    std::cout << "[BENCHMARK] Target: RMSNorm " << type_name << " | Tokens: " << total_tokens << ", Hidden: " << hidden_size << std::endl;

    for (int32_t i = 0; i < warmup_iters; ++i) {
        int32_t buf_idx = i % num_buffers;
        const uint8_t *d_in = reinterpret_cast<const uint8_t *>(d_in_base) + buf_idx * single_input_bytes;
        uint8_t *d_out = reinterpret_cast<uint8_t *>(d_out_base) + buf_idx * single_output_bytes;
        launch_fused_rmsnorm_forward(d_out, d_in, d_gamma, d_gamma_scales, epsilon, total_tokens, hidden_size, data_type, threads_per_block, nullptr);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int32_t i = 0; i < bench_iters; ++i) {
        int32_t buf_idx = i % num_buffers;
        const uint8_t *d_in = reinterpret_cast<const uint8_t *>(d_in_base) + buf_idx * single_input_bytes;
        uint8_t *d_out = reinterpret_cast<uint8_t *>(d_out_base) + buf_idx * single_output_bytes;

        cudaEventRecord(start, nullptr);
        launch_fused_rmsnorm_forward(d_out, d_in, d_gamma, d_gamma_scales, epsilon, total_tokens, hidden_size, data_type, threads_per_block, nullptr);
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

    size_t gamma_size_bytes = hidden_size * sizeof(__nv_bfloat16);
    size_t scales_size_bytes = (data_type == static_cast<int32_t>(DataType::FP4)) ? (hidden_size / 32 * sizeof(float)) : 0;

    double total_bytes_moved = static_cast<double>(single_input_bytes + single_output_bytes + gamma_size_bytes + scales_size_bytes);
    double avg_gb_s = (total_bytes_moved * 1e-9) / (static_cast<double>(avg_time_ms) * 1e-3);

    std::cout << "==========================================================================" << std::endl;
    std::cout << "[RMSNORM BENCHMARK RESULTS - " << type_name << "]" << std::endl;
    std::cout << "  Average Time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "  Median (50%): " << p50 << " ms" << std::endl;
    std::cout << "  Percentile 90%: " << p90 << " ms" << std::endl;
    std::cout << "  Percentile 95%: " << p95 << " ms" << std::endl;
    std::cout << "  Memory Bandwidth Efficiency: " << avg_gb_s << " GB/s" << std::endl;
    std::cout << "==========================================================================" << std::endl;
}

TEST_CASE("GgufRMSNormBenchmark - ComprehensivePerformanceProfile") {
    constexpr int32_t total_tokens = 4096;
    constexpr int32_t hidden_size = 4096;
    constexpr int32_t num_buffers = 8;

    std::vector<__nv_bfloat16> h_gamma(hidden_size, __float2bfloat16(1.0f));
    void *d_gamma = nullptr;
    REQUIRE(cudaMalloc(&d_gamma, hidden_size * sizeof(__nv_bfloat16)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_gamma, h_gamma.data(), hidden_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice) == cudaSuccess);

    void *d_out = nullptr;
    size_t total_output_bytes = total_tokens * hidden_size * sizeof(__nv_bfloat16) * num_buffers;
    REQUIRE(cudaMalloc(&d_out, total_output_bytes) == cudaSuccess);

    {
        size_t single_in_bytes = total_tokens * hidden_size * sizeof(__nv_bfloat16);
        size_t total_in_bytes = single_in_bytes * num_buffers;
        std::vector<__nv_bfloat16> h_in(total_tokens * hidden_size, __float2bfloat16(0.5f));

        void *d_in = nullptr;
        REQUIRE(cudaMalloc(&d_in, total_in_bytes) == cudaSuccess);

        for (int32_t b = 0; b < num_buffers; ++b) {
            uint8_t *dst = reinterpret_cast<uint8_t *>(d_in) + b * single_in_bytes;
            REQUIRE(cudaMemcpy(dst, h_in.data(), single_in_bytes, cudaMemcpyHostToDevice) == cudaSuccess);
        }

        run_rmsnorm_benchmark(static_cast<int32_t>(DataType::BF16), "BF16", total_tokens, hidden_size, d_in, d_gamma, nullptr, d_out, single_in_bytes);
        cudaFree(d_in);
    }

    {
        size_t single_in_bytes = total_tokens * hidden_size * sizeof(uint8_t);
        size_t total_in_bytes = single_in_bytes * num_buffers;
        std::vector<uint8_t> h_in(total_tokens * hidden_size, 0x25);

        void *d_in = nullptr;
        REQUIRE(cudaMalloc(&d_in, total_in_bytes) == cudaSuccess);

        for (int32_t b = 0; b < num_buffers; ++b) {
            uint8_t *dst = reinterpret_cast<uint8_t *>(d_in) + b * single_in_bytes;
            REQUIRE(cudaMemcpy(dst, h_in.data(), single_in_bytes, cudaMemcpyHostToDevice) == cudaSuccess);
        }

        run_rmsnorm_benchmark(static_cast<int32_t>(DataType::FP8), "FP8", total_tokens, hidden_size, d_in, d_gamma, nullptr, d_out, single_in_bytes);
        cudaFree(d_in);
    }

    {
        size_t single_in_bytes = total_tokens * (hidden_size / 2) * sizeof(uint8_t);
        size_t total_in_bytes = single_in_bytes * num_buffers;
        std::vector<uint8_t> h_in(total_tokens * (hidden_size / 2), 0x12);

        void *d_in = nullptr;
        REQUIRE(cudaMalloc(&d_in, total_in_bytes) == cudaSuccess);

        for (int32_t b = 0; b < num_buffers; ++b) {
            uint8_t *dst = reinterpret_cast<uint8_t *>(d_in) + b * single_in_bytes;
            REQUIRE(cudaMemcpy(dst, h_in.data(), single_in_bytes, cudaMemcpyHostToDevice) == cudaSuccess);
        }

        size_t size_scales = (hidden_size / 32) * sizeof(float);
        std::vector<float> h_scales(hidden_size / 32, 1.25f);
        float *d_scales = nullptr;
        REQUIRE(cudaMalloc(&d_scales, size_scales) == cudaSuccess);
        REQUIRE(cudaMemcpy(d_scales, h_scales.data(), size_scales, cudaMemcpyHostToDevice) == cudaSuccess);

        run_rmsnorm_benchmark(static_cast<int32_t>(DataType::FP4), "FP4", total_tokens, hidden_size, d_in, d_gamma, d_scales, d_out, single_in_bytes);
        cudaFree(d_in);
        cudaFree(d_scales);
    }

    cudaFree(d_gamma);
    cudaFree(d_out);
}
