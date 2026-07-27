#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <vector>
#include <string>
#include <iostream>

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
    void *d_out, const size_t single_input_bytes, const L2CacheFlusher &flusher
) {
    constexpr int32_t warmup_iters = 50;
    constexpr int32_t bench_iters = 200;
    constexpr int32_t threads_per_block = 64;
    constexpr float epsilon = 1e-5f;

    std::vector<float> iters_ms(bench_iters);
    const GPUTimer timer;

    for (int32_t i = 0; i < warmup_iters; ++i) {
        launch_fused_rmsnorm_forward(
            d_out, d_in, d_gamma, d_gamma_scales, epsilon,
            total_tokens, hidden_size, data_type, threads_per_block, nullptr
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int32_t i = 0; i < bench_iters; ++i) {
        flusher.flush();

        timer.start();
        launch_fused_rmsnorm_forward(
            d_out, d_in, d_gamma, d_gamma_scales, epsilon,
            total_tokens, hidden_size, data_type, threads_per_block, nullptr
        );
        timer.stop();
        iters_ms[i] = timer.elapsed_ms();
    }

    const size_t single_output_bytes = total_tokens * hidden_size * sizeof(__nv_bfloat16);
    const size_t gamma_size_bytes = hidden_size * sizeof(__nv_bfloat16);
    const size_t scales_size_bytes = (data_type == static_cast<int32_t>(DataType::FP4))
                                         ? (hidden_size / 32 * sizeof(float))
                                         : 0;
    const double total_bytes_moved = static_cast<double>(single_input_bytes + single_output_bytes + gamma_size_bytes + scales_size_bytes);

    BenchmarkReporter::report_performance("RMSNORM", type_name, iters_ms, 0.0, total_bytes_moved);
}

void run_rmsnorm_benchmarks() {
    constexpr int32_t total_tokens = 4096;
    constexpr int32_t hidden_size = 4096;

    L2CacheFlusher flusher;

    const std::vector h_gamma(hidden_size, __float2bfloat16(1.0f));
    const DeviceBuffer d_gamma(h_gamma);
    const DeviceBuffer<__nv_bfloat16> d_out(total_tokens * hidden_size);

    {
        constexpr size_t elements = total_tokens * hidden_size;
        const std::vector h_in(elements, __float2bfloat16(0.5f));
        const DeviceBuffer d_in(h_in);

        run_rmsnorm_benchmark(
            static_cast<int32_t>(DataType::BF16), "BF16", total_tokens, hidden_size,
            d_in.get(), d_gamma.get(), nullptr, d_out.get(), d_in.size_bytes(), flusher
        );
    }

    {
        constexpr size_t elements = total_tokens * hidden_size;
        const std::vector<uint8_t> h_in(elements, 0x25);
        const DeviceBuffer d_in(h_in);

        run_rmsnorm_benchmark(
            static_cast<int32_t>(DataType::FP8), "FP8", total_tokens, hidden_size,
            d_in.get(), d_gamma.get(), nullptr, d_out.get(), d_in.size_bytes(), flusher
        );
    }

    {
        constexpr size_t elements = total_tokens * (hidden_size / 2);
        const std::vector<uint8_t> h_in(elements, 0x12);
        const DeviceBuffer d_in(h_in);

        const std::vector h_scales(hidden_size / 32, 1.25f);
        const DeviceBuffer d_scales(h_scales);

        run_rmsnorm_benchmark(
            static_cast<int32_t>(DataType::FP4), "FP4", total_tokens, hidden_size,
            d_in.get(), d_gamma.get(), d_scales.get(), d_out.get(), d_in.size_bytes(), flusher
        );
    }
}

REGISTER_BENCHMARK("rmsnorm", run_rmsnorm_benchmarks);
