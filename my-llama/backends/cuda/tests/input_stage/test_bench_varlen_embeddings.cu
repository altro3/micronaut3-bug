#include <doctest/doctest.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <ranges>
#include <random>

extern "C" void launch_varlen_embeddings(
    void *out,
    const void *weight,
    const float *weight_scales,
    const uint32_t *tokens,
    const int32_t *seq_offsets,
    const int32_t *block_table,
    int32_t *slot_mapping,
    int32_t max_blocks_per_seq,
    int32_t block_size,
    int32_t total_tokens,
    int32_t out_features,
    int32_t vocab_size,
    int32_t num_seqs,
    int32_t data_type,
    int32_t threads_per_block,
    void *stream_ptr
);

static void run_embeddings_benchmark(
    const int32_t data_type, const std::string &type_name,
    const int32_t total_tokens, const int32_t out_features, const int32_t vocab_size,
    const void *d_weight, const float *d_weight_scales, const uint32_t *d_tokens,
    const int32_t *d_seq_offsets, const int32_t *d_block_table, int32_t *d_slot_mapping,
    int32_t max_blocks, int32_t b_size, int32_t n_seqs,
    void *d_out_base, size_t single_weight_row_bytes
) {
    constexpr int32_t warmup_iters = 50;
    constexpr int32_t bench_iters = 200;
    constexpr int32_t threads_per_block = 128;
    constexpr int32_t num_buffers = 8;

    size_t single_output_bytes = total_tokens * out_features * sizeof(__nv_bfloat16);

    auto launch_helper = [&](void *out_ptr) {
        launch_varlen_embeddings(
            out_ptr, d_weight, d_weight_scales, d_tokens, d_seq_offsets, d_block_table, d_slot_mapping,
            max_blocks, b_size, total_tokens, out_features, vocab_size, n_seqs,
            data_type, threads_per_block, nullptr
        );
    };

    for (int32_t i = 0; i < warmup_iters; ++i) {
        int32_t buf_idx = i % num_buffers;
        uint8_t *d_out = reinterpret_cast<uint8_t *>(d_out_base) + buf_idx * single_output_bytes;
        launch_helper(d_out);
    }
    cudaDeviceSynchronize();

    std::vector<cudaEvent_t> start_events(bench_iters);
    std::vector<cudaEvent_t> stop_events(bench_iters);
    for (int32_t i = 0; i < bench_iters; ++i) {
        cudaEventCreate(&start_events[i]);
        cudaEventCreate(&stop_events[i]);
    }

    for (int32_t i = 0; i < bench_iters; ++i) {
        int32_t buf_idx = i % num_buffers;
        uint8_t *d_out = reinterpret_cast<uint8_t *>(d_out_base) + buf_idx * single_output_bytes;

        cudaEventRecord(start_events[i], nullptr);
        launch_helper(d_out);
        cudaEventRecord(stop_events[i], nullptr);
    }

    cudaDeviceSynchronize();

    std::vector<float> iters_ms(bench_iters);
    float sum_time = 0.0f;
    for (int32_t i = 0; i < bench_iters; ++i) {
        cudaEventElapsedTime(&iters_ms[i], start_events[i], stop_events[i]);
        sum_time += iters_ms[i];
        cudaEventDestroy(start_events[i]);
        cudaEventDestroy(stop_events[i]);
    }

    std::ranges::sort(iters_ms);

    const float avg_time_ms = sum_time / bench_iters;
    const float p50 = iters_ms[static_cast<int32_t>(bench_iters * 0.50)];
    const float p90 = iters_ms[static_cast<int32_t>(bench_iters * 0.90)];
    const float p95 = iters_ms[static_cast<int32_t>(bench_iters * 0.95)];

    size_t total_read_bytes = total_tokens * single_weight_row_bytes;
    if (d_weight_scales != nullptr) {
        total_read_bytes += total_tokens * (out_features / 32 * sizeof(float));
    }
    size_t total_write_bytes = total_tokens * out_features * sizeof(__nv_bfloat16);
    double total_bytes_moved = static_cast<double>(total_read_bytes + total_write_bytes);
    double avg_gb_s = (total_bytes_moved * 1e-9) / (static_cast<double>(avg_time_ms) * 1e-3);

    std::cout << "==========================================================================" << std::endl;
    std::cout << "[EMBEDDINGS BENCHMARK RESULTS - " << type_name << "]" << std::endl;
    std::cout << "  Average Time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "  Median (50%): " << p50 << " ms" << std::endl;
    std::cout << "  Percentile 90%: " << p90 << " ms" << std::endl;
    std::cout << "  Percentile 95%: " << p95 << " ms" << std::endl;
    std::cout << "  Memory Bandwidth Efficiency: " << avg_gb_s << " GB/s" << std::endl;
    std::cout << "==========================================================================" << std::endl;
}

TEST_CASE("VarlenEmbeddingsBenchmark - ComprehensivePerformanceProfile") {
    constexpr int32_t total_tokens = 2048;
    constexpr int32_t out_features = 4096;
    constexpr int32_t vocab_size = 32000;
    constexpr int32_t num_seqs = 4;
    constexpr int32_t block_size = 16;
    constexpr int32_t max_blocks_per_seq = 64;
    constexpr int32_t num_buffers = 8;

    std::mt19937 gen(42);
    std::uniform_int_distribution<uint32_t> token_dist(0, vocab_size - 1);

    std::vector<uint32_t> h_tokens(total_tokens);
    for (int32_t i = 0; i < total_tokens; ++i) h_tokens[i] = token_dist(gen);

    std::vector h_seq_offsets = {0, 512, 1024, 1536, 2048};
    std::vector h_block_table(num_seqs * max_blocks_per_seq, 0);
    for (size_t i = 0; i < h_block_table.size(); ++i) {
        h_block_table[i] = static_cast<int32_t>(i / max_blocks_per_seq * 10 + (i % max_blocks_per_seq));
    }

    std::vector<int32_t> h_slot_mapping(total_tokens);
    for (int32_t bi = 0; bi < num_seqs; ++bi) {
        int32_t start_tok = h_seq_offsets[bi];
        int32_t end_tok = h_seq_offsets[bi + 1];
        for (int32_t t = start_tok; t < end_tok; ++t) {
            int32_t token_idx_in_seq = t - start_tok;
            int32_t block_logical_idx = token_idx_in_seq / block_size;
            int32_t block_offset = token_idx_in_seq % block_size;
            int32_t physical_block = h_block_table[bi * max_blocks_per_seq + block_logical_idx];
            h_slot_mapping[t] = physical_block * block_size + block_offset;
        }
    }

    uint32_t *d_tokens = nullptr;
    int32_t *d_seq_offsets = nullptr;
    int32_t *d_block_table = nullptr;
    int32_t *d_slot_mapping = nullptr;
    float *d_weight_scales = nullptr;

    REQUIRE(cudaMalloc(&d_tokens, total_tokens * sizeof(uint32_t)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_seq_offsets, h_seq_offsets.size() * sizeof(int32_t)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_block_table, h_block_table.size() * sizeof(int32_t)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_slot_mapping, total_tokens * sizeof(int32_t)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_weight_scales, vocab_size * (out_features / 32) * sizeof(float)) == cudaSuccess);

    REQUIRE(cudaMemcpy(d_tokens, h_tokens.data(), total_tokens * sizeof(uint32_t), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_seq_offsets, h_seq_offsets.data(), h_seq_offsets.size() * sizeof(int32_t), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_block_table, h_block_table.data(), h_block_table.size() * sizeof(int32_t), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_slot_mapping, h_slot_mapping.data(), total_tokens * sizeof(int32_t), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemset(d_weight_scales, 0, vocab_size * (out_features / 32) * sizeof(float)) == cudaSuccess);

    void *d_out = nullptr;
    size_t total_output_bytes = total_tokens * out_features * sizeof(__nv_bfloat16) * num_buffers;
    REQUIRE(cudaMalloc(&d_out, total_output_bytes) == cudaSuccess);

    {
        size_t total_weight_elements = static_cast<size_t>(vocab_size) * out_features;
        size_t weight_size_bytes = total_weight_elements * sizeof(__nv_bfloat16);
        void *d_weight = nullptr;
        REQUIRE(cudaMalloc(&d_weight, weight_size_bytes) == cudaSuccess);
        REQUIRE(cudaMemset(d_weight, 0, weight_size_bytes) == cudaSuccess);

        run_embeddings_benchmark(0, "BF16", total_tokens, out_features, vocab_size, d_weight, nullptr, d_tokens, d_seq_offsets, d_block_table, d_slot_mapping, max_blocks_per_seq, block_size, num_seqs, d_out, out_features * sizeof(__nv_bfloat16));
        cudaFree(d_weight);
    }

    {
        size_t total_weight_elements = static_cast<size_t>(vocab_size) * out_features;
        size_t weight_size_bytes = total_weight_elements * sizeof(uint8_t);
        void *d_weight = nullptr;
        REQUIRE(cudaMalloc(&d_weight, weight_size_bytes) == cudaSuccess);
        REQUIRE(cudaMemset(d_weight, 0, weight_size_bytes) == cudaSuccess);

        run_embeddings_benchmark(1, "FP8", total_tokens, out_features, vocab_size, d_weight, d_weight_scales, d_tokens, d_seq_offsets, d_block_table, d_slot_mapping, max_blocks_per_seq, block_size, num_seqs, d_out, out_features * sizeof(uint8_t));
        cudaFree(d_weight);
    }

    {
        size_t total_weight_elements = static_cast<size_t>(vocab_size) * (out_features / 2);
        size_t weight_size_bytes = total_weight_elements * sizeof(uint8_t);
        void *d_weight = nullptr;
        REQUIRE(cudaMalloc(&d_weight, weight_size_bytes) == cudaSuccess);
        REQUIRE(cudaMemset(d_weight, 0, weight_size_bytes) == cudaSuccess);

        run_embeddings_benchmark(2, "FP4", total_tokens, out_features, vocab_size, d_weight, d_weight_scales, d_tokens, d_seq_offsets, d_block_table, d_slot_mapping, max_blocks_per_seq, block_size, num_seqs, d_out, out_features / 2);
        cudaFree(d_weight);
    }

    cudaFree(d_tokens);
    cudaFree(d_seq_offsets);
    cudaFree(d_block_table);
    cudaFree(d_slot_mapping);
    cudaFree(d_weight_scales);
    cudaFree(d_out);
}
