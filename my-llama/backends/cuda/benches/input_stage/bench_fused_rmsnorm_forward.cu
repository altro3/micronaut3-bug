#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <ranges>

#include "data_types.h"
#include "../bench_utils.cuh"

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
    const void *d_in, const void *d_gamma, const float *d_gamma_scales,
    void *d_out, const size_t single_input_bytes, L2CacheFlusher &flusher
) {
    constexpr int32_t warmup_iters = 50;
    constexpr int32_t bench_iters = 200;
    constexpr int32_t threads_per_block = 64;
    constexpr float epsilon = 1e-5f;

    std::vector<float> iters_ms(bench_iters);
    const GPUTimer timer;

    const size_t single_output_bytes = total_tokens * hidden_size * sizeof(__nv_bfloat16);

    std::cout << "[BENCHMARK] Target: RMSNorm " << type_name
            << " | Tokens: " << total_tokens
            << ", Hidden: " << hidden_size << std::endl;

    for (int32_t i = 0; i < warmup_iters; ++i) {
        launch_fused_rmsnorm_forward(
            d_out, d_in, d_gamma, d_gamma_scales, epsilon,
            total_tokens, hidden_size, data_type, threads_per_block, nullptr
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    float total_time_ms = 0.0f;

    for (int32_t i = 0; i < bench_iters; ++i) {
        flusher.flush();

        timer.start();
        launch_fused_rmsnorm_forward(
            d_out, d_in, d_gamma, d_gamma_scales, epsilon,
            total_tokens, hidden_size, data_type, threads_per_block, nullptr
        );
        timer.stop();

        const float ms = timer.elapsed_ms();
        iters_ms[i] = ms;
        total_time_ms += ms;
    }

    std::ranges::sort(iters_ms);

    const float avg_time_ms = total_time_ms / bench_iters;
    const float p50 = iters_ms[static_cast<int32_t>(bench_iters * 0.50)];
    const float p90 = iters_ms[static_cast<int32_t>(bench_iters * 0.90)];
    const float p95 = iters_ms[static_cast<int32_t>(bench_iters * 0.95)];

    const size_t gamma_size_bytes = hidden_size * sizeof(__nv_bfloat16);
    const size_t scales_size_bytes = data_type == static_cast<int32_t>(DataType::FP4)
                                         ? hidden_size / 32 * sizeof(float)
                                         : 0;

    const double total_bytes_moved = static_cast<double>(single_input_bytes + single_output_bytes + gamma_size_bytes + scales_size_bytes);
    const double avg_gb_s = total_bytes_moved * 1e-9 / (static_cast<double>(avg_time_ms) * 1e-3);

    std::cout << "==========================================================================" << std::endl;
    std::cout << "[RMSNORM BENCHMARK RESULTS - " << type_name << "]" << std::endl;
    std::cout << "  Average Time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "  Median (50%): " << p50 << " ms" << std::endl;
    std::cout << "  Percentile 90%: " << p90 << " ms" << std::endl;
    std::cout << "  Percentile 95%: " << p95 << " ms" << std::endl;
    std::cout << "  Memory Bandwidth Efficiency: " << avg_gb_s << " GB/s" << std::endl;
    std::cout << "==========================================================================" << std::endl;
}

extern "C" {
void run_rmsnorm_benchmarks() {
    constexpr int32_t total_tokens = 4096;
    constexpr int32_t hidden_size = 4096;

    L2CacheFlusher flusher;

    const std::vector h_gamma(hidden_size, __float2bfloat16(1.0f));
    void *d_gamma = nullptr;
    CUDA_CHECK(cudaMalloc(&d_gamma, hidden_size * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), hidden_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    void *d_out = nullptr;
    constexpr size_t total_output_bytes = total_tokens * hidden_size * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMalloc(&d_out, total_output_bytes));

    {
        constexpr size_t single_in_bytes = total_tokens * hidden_size * sizeof(__nv_bfloat16);
        const std::vector h_in(total_tokens * hidden_size, __float2bfloat16(0.5f));

        void *d_in = nullptr;
        CUDA_CHECK(cudaMalloc(&d_in, single_in_bytes));
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), single_in_bytes, cudaMemcpyHostToDevice));

        run_rmsnorm_benchmark(
            static_cast<int32_t>(DataType::BF16), "BF16", total_tokens, hidden_size,
            d_in, d_gamma, nullptr, d_out, single_in_bytes, flusher
        );
        CUDA_CHECK(cudaFree(d_in));
    }

    {
        constexpr size_t single_in_bytes = total_tokens * hidden_size * sizeof(uint8_t);
        const std::vector<uint8_t> h_in(total_tokens * hidden_size, 0x25);

        void *d_in = nullptr;
        CUDA_CHECK(cudaMalloc(&d_in, single_in_bytes));
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), single_in_bytes, cudaMemcpyHostToDevice));

        run_rmsnorm_benchmark(
            static_cast<int32_t>(DataType::FP8), "FP8", total_tokens, hidden_size,
            d_in, d_gamma, nullptr, d_out, single_in_bytes, flusher
        );
        CUDA_CHECK(cudaFree(d_in));
    }

    {
        constexpr size_t single_in_bytes = total_tokens * (hidden_size / 2) * sizeof(uint8_t);
        const std::vector<uint8_t> h_in(total_tokens * (hidden_size / 2), 0x12);

        void *d_in = nullptr;
        CUDA_CHECK(cudaMalloc(&d_in, single_in_bytes));
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), single_in_bytes, cudaMemcpyHostToDevice));

        constexpr size_t size_scales = hidden_size / 32 * sizeof(float);
        const std::vector h_scales(hidden_size / 32, 1.25f);
        float *d_scales = nullptr;
        CUDA_CHECK(cudaMalloc(&d_scales, size_scales));
        CUDA_CHECK(cudaMemcpy(d_scales, h_scales.data(), size_scales, cudaMemcpyHostToDevice));

        run_rmsnorm_benchmark(
            static_cast<int32_t>(DataType::FP4), "FP4", total_tokens, hidden_size,
            d_in, d_gamma, d_scales, d_out, single_in_bytes, flusher
        );
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_scales));
    }

    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_out));
}
}

REGISTER_BENCHMARK("rmsnorm", run_rmsnorm_benchmarks);
