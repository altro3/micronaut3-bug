#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include "input_stage/fused_multimodal_projection.cuh"
#include <vector>
#include <string>
#include <iostream>
#include <ranges>
#include <cstdlib>

#include "data_types.h"
#include "../bench_utils.cuh"

extern "C" {
void launch_fused_multimodal_projection(
    const void ** __restrict__ host_ptr_A,
    const void ** __restrict__ host_ptr_B,
    void ** __restrict__ host_ptr_D,
    const float * __restrict__ bias,
    const float * __restrict__ weight_scales,
    const int32_t * __restrict__ host_problem_shapes,
    int32_t num_segments,
    int32_t vision_hidden_size,
    int32_t text_hidden_size,
    int32_t tp_rank,
    int32_t tp_size,
    int32_t data_type,
    void *stream_ptr
);
}

static void run_projection_benchmark(
    const int32_t data_type, const std::string &type_name,
    const int32_t total_tokens, const int32_t vision_hidden_size, const int32_t text_hidden_size,
    const void **h_inputs, const void **h_weights, void **h_outputs,
    const float *d_bias, const float *d_scales,
    const int32_t *h_shapes_array, const int32_t num_segments, size_t single_weight_bytes,
    const L2CacheFlusher &flusher
) {
    constexpr int32_t warmup_iters = 20;
    constexpr int32_t bench_iters = 100;

    std::vector<float> iters_ms(bench_iters);
    const GPUTimer timer;

    auto launch_helper = [&] {
        launch_fused_multimodal_projection(
            h_inputs, h_weights, h_outputs, d_bias, d_scales, h_shapes_array,
            num_segments, vision_hidden_size, text_hidden_size, 0, 1, data_type, nullptr
        );
    };

    for (int32_t i = 0; i < warmup_iters; ++i) {
        launch_helper();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int32_t i = 0; i < bench_iters; ++i) {
        flusher.flush();

        timer.start();
        launch_helper();
        timer.stop();
        iters_ms[i] = timer.elapsed_ms();
    }

    const double operations = 2.0 * static_cast<double>(total_tokens) * static_cast<double>(vision_hidden_size) * static_cast<double>(text_hidden_size);

    const size_t read_A_bytes = total_tokens * vision_hidden_size * 2;
    const size_t read_B_bytes = num_segments * single_weight_bytes;
    const size_t write_C_bytes = total_tokens * text_hidden_size * 2;
    const double total_bytes_moved = static_cast<double>(read_A_bytes + read_B_bytes + write_C_bytes);

    BenchmarkReporter::report_performance("MULTIMODAL PROJECTION", type_name, iters_ms, operations, total_bytes_moved);
}

void run_multimodal_projection_benchmarks() {
    constexpr int32_t num_segments = 4;
    constexpr int32_t tokens_per_segment = 2048;
    constexpr int32_t total_tokens = num_segments * tokens_per_segment;
    constexpr int32_t vision_hidden_size = 4096;
    constexpr int32_t text_hidden_size = 4096;

    if (num_segments > MAX_SEGMENTS) {
        std::cerr << "Error: num_segments exceeds MAX_SEGMENTS" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    const L2CacheFlusher flusher;

    std::vector<int32_t> h_shapes_array(num_segments * 3);
    for (int32_t i = 0; i < num_segments; ++i) {
        h_shapes_array[i * 3 + 0] = tokens_per_segment;
        h_shapes_array[i * 3 + 1] = text_hidden_size;
        h_shapes_array[i * 3 + 2] = vision_hidden_size;
    }

    size_t input_bytes_per_seg = tokens_per_segment * vision_hidden_size * 2;
    size_t output_bytes_per_seg = tokens_per_segment * text_hidden_size * 2;

    size_t weight_bf16_bytes = text_hidden_size * vision_hidden_size * 2;
    size_t weight_fp8_bytes = text_hidden_size * vision_hidden_size * 1;
    size_t weight_fp4_bytes = text_hidden_size * vision_hidden_size / 2;

    std::vector<DeviceBuffer<uint8_t> > allocated_inputs;
    std::vector<DeviceBuffer<uint8_t> > allocated_outputs;
    std::vector<DeviceBuffer<uint8_t> > allocated_weights_bf16;
    std::vector<DeviceBuffer<uint8_t> > allocated_weights_fp8;
    std::vector<DeviceBuffer<uint8_t> > allocated_weights_fp4;

    allocated_inputs.reserve(num_segments);
    allocated_outputs.reserve(num_segments);
    allocated_weights_bf16.reserve(num_segments);
    allocated_weights_fp8.reserve(num_segments);
    allocated_weights_fp4.reserve(num_segments);

    for (int32_t i = 0; i < num_segments; ++i) {
        allocated_inputs.emplace_back(input_bytes_per_seg, 0);
        allocated_outputs.emplace_back(output_bytes_per_seg, 0);
        allocated_weights_bf16.emplace_back(weight_bf16_bytes, 0);
        allocated_weights_fp8.emplace_back(weight_fp8_bytes, 0);
        allocated_weights_fp4.emplace_back(weight_fp4_bytes, 0);
    }

    const DeviceBuffer<float> d_bias(text_hidden_size, 0);
    const DeviceBuffer<float> d_scales(num_segments, 0);

    std::vector<const void *> host_inputs_ptr(num_segments);
    std::vector<void *> host_outputs_ptr(num_segments);
    std::vector<const void *> host_weights_ptr(num_segments);

    for (int32_t i = 0; i < num_segments; ++i) {
        host_inputs_ptr[i] = allocated_inputs[i].get_const_void();
        host_outputs_ptr[i] = allocated_outputs[i].get_void();
    }

    {
        for (int32_t i = 0; i < num_segments; ++i) {
            host_weights_ptr[i] = allocated_weights_bf16[i].get_const_void();
        }
        run_projection_benchmark(
            static_cast<int32_t>(DataType::BF16), "BF16", total_tokens, vision_hidden_size, text_hidden_size,
            host_inputs_ptr.data(), host_weights_ptr.data(), host_outputs_ptr.data(),
            d_bias.get(), nullptr, h_shapes_array.data(), num_segments, weight_bf16_bytes,
            flusher
        );
    }
    {
        for (int32_t i = 0; i < num_segments; ++i) {
            host_weights_ptr[i] = allocated_weights_fp8[i].get_const_void();
        }
        run_projection_benchmark(
            static_cast<int32_t>(DataType::FP8), "FP8", total_tokens, vision_hidden_size, text_hidden_size,
            host_inputs_ptr.data(), host_weights_ptr.data(), host_outputs_ptr.data(),
            d_bias.get(), d_scales.get(), h_shapes_array.data(), num_segments, weight_fp8_bytes,
            flusher
        );
    }
    {
        for (int32_t i = 0; i < num_segments; ++i) {
            host_weights_ptr[i] = allocated_weights_fp4[i].get_const_void();
        }
        run_projection_benchmark(
            static_cast<int32_t>(DataType::FP4), "FP4", total_tokens, vision_hidden_size, text_hidden_size,
            host_inputs_ptr.data(), host_weights_ptr.data(), host_outputs_ptr.data(),
            d_bias.get(), d_scales.get(), h_shapes_array.data(), num_segments, weight_fp4_bytes,
            flusher
        );
    }
}

REGISTER_BENCHMARK("projection", run_multimodal_projection_benchmarks);
