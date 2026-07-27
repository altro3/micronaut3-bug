#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <ranges>
#include <random>

#include "data_types.h"
#include "../bench_utils.cuh"

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
    const int32_t max_blocks, const int32_t b_size, const int32_t n_seqs,
    void *d_out, const size_t single_weight_row_bytes, L2CacheFlusher &flusher
) {
    constexpr int32_t warmup_iters = 50;
    constexpr int32_t bench_iters = 200;
    constexpr int32_t threads_per_block = 128;

    std::vector<float> iters_ms(bench_iters);
    const GPUTimer timer;

    auto launch_helper = [&](void *out_ptr) {
        launch_varlen_embeddings(
            out_ptr, d_weight, d_weight_scales, d_tokens, d_seq_offsets, d_block_table, d_slot_mapping,
            max_blocks, b_size, total_tokens, out_features, vocab_size, n_seqs,
            data_type, threads_per_block, nullptr
        );
    };

    for (int32_t i = 0; i < warmup_iters; ++i) {
        launch_helper(d_out);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    float total_time_ms = 0.0f;

    for (int32_t i = 0; i < bench_iters; ++i) {
        flusher.flush();

        timer.start();
        launch_helper(d_out);
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

    size_t total_read_bytes = total_tokens * single_weight_row_bytes;
    if (d_weight_scales != nullptr) {
        total_read_bytes += total_tokens * (out_features / 32 * sizeof(float));
    }
    const size_t total_write_bytes = total_tokens * out_features * sizeof(__nv_bfloat16);
    const double total_bytes_moved = static_cast<double>(total_read_bytes + total_write_bytes);
    const double avg_gb_s = (total_bytes_moved * 1e-9) / (static_cast<double>(avg_time_ms) * 1e-3);

    std::cout << "==========================================================================" << std::endl;
    std::cout << "[EMBEDDINGS BENCHMARK RESULTS - " << type_name << "]" << std::endl;
    std::cout << "  Average Time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "  Median (50%): " << p50 << " ms" << std::endl;
    std::cout << "  Percentile 90%: " << p90 << " ms" << std::endl;
    std::cout << "  Percentile 95%: " << p95 << " ms" << std::endl;
    std::cout << "  Memory Bandwidth Efficiency: " << avg_gb_s << " GB/s" << std::endl;
    std::cout << "==========================================================================" << std::endl;
}

void run_varlen_embeddings_benchmarks() {
    constexpr int32_t total_tokens = 16384;
    constexpr int32_t out_features = 4096;
    constexpr int32_t vocab_size = 256000;
    constexpr int32_t num_seqs = 4;
    constexpr int32_t block_size = 16;
    constexpr int32_t max_blocks_per_seq = 64;

    std::mt19937 gen(42);
    std::uniform_int_distribution<uint32_t> token_dist(0, vocab_size - 1);

    std::vector<uint32_t> h_tokens(total_tokens);
    for (int32_t i = 0; i < total_tokens; ++i) {
        h_tokens[i] = token_dist(gen);
    }

    const std::vector h_seq_offsets = {0, 512, 1024, 1536, 2048};
    std::vector h_block_table(num_seqs * max_blocks_per_seq, 0);
    for (size_t i = 0; i < h_block_table.size(); ++i) {
        h_block_table[i] = static_cast<int32_t>(i / max_blocks_per_seq * 10 + (i % max_blocks_per_seq));
    }

    std::vector<int32_t> h_slot_mapping(total_tokens);
    for (int32_t bi = 0; bi < num_seqs; ++bi) {
        const int32_t start_tok = h_seq_offsets[bi];
        const int32_t end_tok = h_seq_offsets[bi + 1];
        for (int32_t t = start_tok; t < end_tok; ++t) {
            const int32_t token_idx_in_seq = t - start_tok;
            const int32_t block_logical_idx = token_idx_in_seq / block_size;
            const int32_t block_offset = token_idx_in_seq % block_size;
            const int32_t physical_block = h_block_table[bi * max_blocks_per_seq + block_logical_idx];
            h_slot_mapping[t] = physical_block * block_size + block_offset;
        }
    }

    uint32_t *d_tokens = nullptr;
    int32_t *d_seq_offsets = nullptr;
    int32_t *d_block_table = nullptr;
    int32_t *d_slot_mapping = nullptr;
    float *d_weight_scales = nullptr;

    CUDA_CHECK(cudaMalloc(&d_tokens, total_tokens * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_seq_offsets, h_seq_offsets.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_block_table, h_block_table.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_slot_mapping, total_tokens * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_weight_scales, vocab_size * (out_features / 32) * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens.data(), total_tokens * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seq_offsets, h_seq_offsets.data(), h_seq_offsets.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_block_table, h_block_table.data(), h_block_table.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_slot_mapping, h_slot_mapping.data(), total_tokens * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_weight_scales, 0, vocab_size * (out_features / 32) * sizeof(float)));

    void *d_out = nullptr;
    constexpr size_t total_output_bytes = total_tokens * out_features * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMalloc(&d_out, total_output_bytes));

    L2CacheFlusher flusher;

    {
        constexpr size_t total_weight_elements = static_cast<size_t>(vocab_size) * out_features;
        constexpr size_t weight_size_bytes = total_weight_elements * sizeof(__nv_bfloat16);
        void *d_weight = nullptr;
        CUDA_CHECK(cudaMalloc(&d_weight, weight_size_bytes));
        CUDA_CHECK(cudaMemset(d_weight, 0, weight_size_bytes));

        run_embeddings_benchmark(
            static_cast<int32_t>(DataType::BF16), "BF16", total_tokens, out_features, vocab_size,
            d_weight, nullptr, d_tokens, d_seq_offsets, d_block_table, d_slot_mapping,
            max_blocks_per_seq, block_size, num_seqs, d_out, out_features * sizeof(__nv_bfloat16), flusher
        );
        CUDA_CHECK(cudaFree(d_weight));
    }

    {
        constexpr size_t total_weight_elements = static_cast<size_t>(vocab_size) * out_features;
        constexpr size_t weight_size_bytes = total_weight_elements * sizeof(uint8_t);
        void *d_weight = nullptr;
        CUDA_CHECK(cudaMalloc(&d_weight, weight_size_bytes));
        CUDA_CHECK(cudaMemset(d_weight, 0, weight_size_bytes));

        run_embeddings_benchmark(
            static_cast<int32_t>(DataType::FP8), "FP8", total_tokens, out_features, vocab_size,
            d_weight, d_weight_scales, d_tokens, d_seq_offsets, d_block_table, d_slot_mapping,
            max_blocks_per_seq, block_size, num_seqs, d_out, out_features * sizeof(uint8_t), flusher
        );
        CUDA_CHECK(cudaFree(d_weight));
    }

    {
        constexpr size_t total_weight_elements = static_cast<size_t>(vocab_size) * (out_features / 2);
        constexpr size_t weight_size_bytes = total_weight_elements * sizeof(uint8_t);
        void *d_weight = nullptr;
        CUDA_CHECK(cudaMalloc(&d_weight, weight_size_bytes));
        CUDA_CHECK(cudaMemset(d_weight, 0, weight_size_bytes));

        run_embeddings_benchmark(
            static_cast<int32_t>(DataType::FP4), "FP4", total_tokens, out_features, vocab_size,
            d_weight, d_weight_scales, d_tokens, d_seq_offsets, d_block_table, d_slot_mapping,
            max_blocks_per_seq, block_size, num_seqs, d_out, out_features / 2, flusher
        );
        CUDA_CHECK(cudaFree(d_weight));
    }

    CUDA_CHECK(cudaFree(d_tokens));
    CUDA_CHECK(cudaFree(d_seq_offsets));
    CUDA_CHECK(cudaFree(d_block_table));
    CUDA_CHECK(cudaFree(d_slot_mapping));
    CUDA_CHECK(cudaFree(d_weight_scales));
    CUDA_CHECK(cudaFree(d_out));
}

REGISTER_BENCHMARK("embeddings", run_varlen_embeddings_benchmarks);
