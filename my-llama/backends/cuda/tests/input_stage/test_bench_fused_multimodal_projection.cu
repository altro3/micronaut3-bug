#include <doctest/doctest.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include "input_stage/fused_multimodal_projection.cuh"
#include <vector>
#include <string>
#include <iostream>
#include <ranges>

// Перечисление для явного соответствия типов данных в лончере
enum class DataType : int32_t {
    BF16 = 0,
    FP8 = 1,
    FP4 = 2
};

extern "C" {
void launch_fused_multimodal_projection(
    const void ** __restrict__ host_ptr_A,
    const void ** __restrict__ host_ptr_B,
    void ** __restrict__ host_ptr_D,
    const float * __restrict__ bias,
    const float * __restrict__ weight_scales,
    const int32_t * __restrict__ host_problem_shapes,
    const int32_t num_segments,
    const int32_t vision_hidden_size,
    const int32_t text_hidden_size,
    const int32_t tp_rank,
    const int32_t tp_size,
    int32_t data_type,
    void *stream_ptr
);
}

static void run_projection_benchmark(
    const int32_t data_type, const std::string &type_name,
    const int32_t total_tokens, const int32_t vision_hidden_size, const int32_t text_hidden_size,
    const void **h_inputs, const void **h_weights, void **h_outputs,
    const float *d_bias, const float *d_scales,
    const int32_t *h_shapes_array, const int32_t num_segments, size_t single_weight_bytes
) {
    constexpr int32_t warmup_iters = 20;
    constexpr int32_t bench_iters = 100;

    auto launch_helper = [&]() {
        launch_fused_multimodal_projection(
            h_inputs, h_weights, h_outputs, d_bias, d_scales, h_shapes_array,
            num_segments, vision_hidden_size, text_hidden_size, 0, 1, data_type, nullptr
        );
    };

    // 1. Честный прогрев
    for (int32_t i = 0; i < warmup_iters; ++i) {
        launch_helper();
    }
    cudaDeviceSynchronize();

    // 2. Создаем одну пару эвентов один раз
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_time_ms = 0.0f;

    // 3. Замеряем каждую итерацию изолированно, давая GPU завершить работу
    for (int32_t i = 0; i < bench_iters; ++i) {
        cudaEventRecord(start, nullptr);
        launch_helper();
        cudaEventRecord(stop, nullptr);
        cudaEventSynchronize(stop); // Ждем физического окончания выполнения на GPU

        float iter_ms = 0.0f;
        cudaEventElapsedTime(&iter_ms, start, stop);
        total_time_ms += iter_ms;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    const float avg_time_ms = total_time_ms / bench_iters;

    // Расчет честных TFLOPs
    double operations = 2.0 * static_cast<double>(total_tokens) * static_cast<double>(vision_hidden_size) * static_cast<double>(text_hidden_size);
    double avg_tflops = operations * 1e-12 / (static_cast<double>(avg_time_ms) * 1e-3);

    // Расчет реального объема переданных данных (в байтах)
    size_t read_A_bytes = total_tokens * vision_hidden_size * 2; // BF16
    size_t read_B_bytes = num_segments * single_weight_bytes; // Веса читаются из глобальной памяти
    size_t write_C_bytes = total_tokens * text_hidden_size * 2; // BF16

    double total_bytes_moved = static_cast<double>(read_A_bytes + read_B_bytes + write_C_bytes);
    double avg_gb_s = total_bytes_moved * 1e-9 / (static_cast<double>(avg_time_ms) * 1e-3);

    std::cout << "==========================================================================" << std::endl;
    std::cout << "[MULTIMODAL PROJECTION BENCHMARK RESULTS - " << type_name << "]" << std::endl;
    std::cout << "  Real Average Time:   " << avg_time_ms << " ms" << std::endl;
    std::cout << "  Real Compute Perf:   " << avg_tflops << " TFLOPs/s" << std::endl;
    std::cout << "  Real Bandwidth:      " << avg_gb_s << " GB/s" << std::endl;
    std::cout << "==========================================================================" << std::endl;
}

TEST_CASE("MultimodalProjectionBenchmark - PerformanceProfile") {
    constexpr int32_t num_segments = 4;
    constexpr int32_t tokens_per_segment = 2048;
    constexpr int32_t total_tokens = num_segments * tokens_per_segment;
    constexpr int32_t vision_hidden_size = 4096;
    constexpr int32_t text_hidden_size = 4096;

    REQUIRE(num_segments <= MAX_SEGMENTS);

    std::vector<int32_t> h_shapes_array(num_segments * 3);
    for (int32_t i = 0; i < num_segments; ++i) {
        h_shapes_array[i * 3 + 0] = tokens_per_segment;
        h_shapes_array[i * 3 + 1] = text_hidden_size;
        h_shapes_array[i * 3 + 2] = vision_hidden_size;
    }

    size_t input_bytes = tokens_per_segment * vision_hidden_size * 2;
    size_t output_bytes = tokens_per_segment * text_hidden_size * 2;

    size_t weight_bf16_bytes = text_hidden_size * vision_hidden_size * 2;
    size_t weight_fp8_bytes = text_hidden_size * vision_hidden_size * 1;
    size_t weight_fp4_bytes = text_hidden_size * vision_hidden_size / 2;

    std::vector<void *> allocated_inputs(num_segments);
    std::vector<void *> allocated_outputs(num_segments);
    std::vector<void *> allocated_weights_bf16(num_segments);
    std::vector<void *> allocated_weights_fp8(num_segments);
    std::vector<void *> allocated_weights_fp4(num_segments);

    for (int32_t i = 0; i < num_segments; ++i) {
        REQUIRE(cudaMalloc(&allocated_inputs[i], input_bytes) == cudaSuccess);
        REQUIRE(cudaMalloc(&allocated_outputs[i], output_bytes) == cudaSuccess);
        REQUIRE(cudaMalloc(&allocated_weights_bf16[i], weight_bf16_bytes) == cudaSuccess);
        REQUIRE(cudaMalloc(&allocated_weights_fp8[i], weight_fp8_bytes) == cudaSuccess);
        REQUIRE(cudaMalloc(&allocated_weights_fp4[i], weight_fp4_bytes) == cudaSuccess);

        REQUIRE(cudaMemset(allocated_inputs[i], 0, input_bytes) == cudaSuccess);
        REQUIRE(cudaMemset(allocated_outputs[i], 0, output_bytes) == cudaSuccess);
        REQUIRE(cudaMemset(allocated_weights_bf16[i], 0, weight_bf16_bytes) == cudaSuccess);
        REQUIRE(cudaMemset(allocated_weights_fp8[i], 0, weight_fp8_bytes) == cudaSuccess);
        REQUIRE(cudaMemset(allocated_weights_fp4[i], 0, weight_fp4_bytes) == cudaSuccess);
    }

    float *d_bias = nullptr, *d_scales = nullptr;
    REQUIRE(cudaMalloc(&d_bias, text_hidden_size * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_scales, num_segments * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMemset(d_bias, 0, text_hidden_size * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMemset(d_scales, 0, num_segments * sizeof(float)) == cudaSuccess);

    std::vector<const void *> host_inputs_ptr(num_segments);
    std::vector<void *> host_outputs_ptr(num_segments);
    std::vector<const void *> host_weights_ptr(num_segments);

    for (int32_t i = 0; i < num_segments; ++i) {
        host_inputs_ptr[i] = allocated_inputs[i];
        host_outputs_ptr[i] = allocated_outputs[i];
    }

    {
        for (int32_t i = 0; i < num_segments; ++i) {
            host_weights_ptr[i] = allocated_weights_bf16[i];
        }
        run_projection_benchmark(
            static_cast<int32_t>(DataType::BF16), "BF16", total_tokens, vision_hidden_size, text_hidden_size,
            host_inputs_ptr.data(), host_weights_ptr.data(), host_outputs_ptr.data(),
            d_bias, nullptr, h_shapes_array.data(), num_segments, weight_bf16_bytes
        );
    }
    {
        for (int32_t i = 0; i < num_segments; ++i) {
            host_weights_ptr[i] = allocated_weights_fp8[i];
        }
        run_projection_benchmark(
            static_cast<int32_t>(DataType::FP8), "FP8", total_tokens, vision_hidden_size, text_hidden_size,
            host_inputs_ptr.data(), host_weights_ptr.data(), host_outputs_ptr.data(),
            d_bias, d_scales, h_shapes_array.data(), num_segments, weight_fp8_bytes
        );
    }
    {
        for (int32_t i = 0; i < num_segments; ++i) {
            host_weights_ptr[i] = allocated_weights_fp4[i];
        }
        run_projection_benchmark(
            static_cast<int32_t>(DataType::FP4), "FP4", total_tokens, vision_hidden_size, text_hidden_size,
            host_inputs_ptr.data(), host_weights_ptr.data(), host_outputs_ptr.data(),
            d_bias, d_scales, h_shapes_array.data(), num_segments, weight_fp4_bytes
        );
    }

    for (int32_t i = 0; i < num_segments; ++i) {
        cudaFree(allocated_inputs[i]);
        cudaFree(allocated_outputs[i]);
        cudaFree(allocated_weights_bf16[i]);
        cudaFree(allocated_weights_fp8[i]);
        cudaFree(allocated_weights_fp4[i]);
    }
    cudaFree(d_bias);
    cudaFree(d_scales);
}
