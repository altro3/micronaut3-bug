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
    int32_t max_blocks, int32_t b_size, int32_t n_seqs,
    void *d_out, size_t single_weight_row_bytes, L2CacheFlusher &flusher
) {
    constexpr int32_t warmup_iters = 50;
    constexpr int32_t bench_iters = 200;
    constexpr int32_t threads_per_block = 128;

    std::vector<float> iters_ms(bench_iters);
    GPUTimer timer;

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

    for (int32_t i = 0; i < bench_iters; ++i) {
        flusher.flush();

        timer.start();
        launch_helper(d_out);
        timer.stop();
        iters_ms[i] = timer.elapsed_ms();
    }

    size_t total_read_bytes = total_tokens * single_weight_row_bytes;
    if (d_weight_scales != nullptr) {
        total_read_bytes += total_tokens * (out_features / 32 * sizeof(float));
    }
    size_t total_write_bytes = total_tokens * out_features * sizeof(__nv_bfloat16);
    double total_bytes_moved = static_cast<double>(total_read_bytes + total_write_bytes);

    BenchmarkReporter::report_performance("EMBEDDINGS", type_name, iters_ms, 0.0, total_bytes_moved);
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

    DeviceBuffer d_tokens(h_tokens);
    DeviceBuffer d_seq_offsets(h_seq_offsets);
    DeviceBuffer d_block_table(h_block_table);
    DeviceBuffer d_slot_mapping(h_slot_mapping);
    DeviceBuffer<float> d_weight_scales(vocab_size * (out_features / 32), 0);
    DeviceBuffer<__nv_bfloat16> d_out(total_tokens * out_features);

    L2CacheFlusher flusher;

    {
        size_t total_weight_elements = static_cast<size_t>(vocab_size) * out_features;
        DeviceBuffer<__nv_bfloat16> d_weight(total_weight_elements, 0);

        run_embeddings_benchmark(
            static_cast<int32_t>(DataType::BF16), "BF16", total_tokens, out_features, vocab_size,
            d_weight.get(), nullptr, d_tokens.get(), d_seq_offsets.get(), d_block_table.get(), d_slot_mapping.get(),
            max_blocks_per_seq, block_size, num_seqs, d_out.get(), out_features * sizeof(__nv_bfloat16), flusher
        );
    }

    {
        size_t total_weight_elements = static_cast<size_t>(vocab_size) * out_features;
        DeviceBuffer<uint8_t> d_weight(total_weight_elements, 0);

        run_embeddings_benchmark(
            static_cast<int32_t>(DataType::FP8), "FP8", total_tokens, out_features, vocab_size,
            d_weight.get(), d_weight_scales.get(), d_tokens.get(), d_seq_offsets.get(), d_block_table.get(), d_slot_mapping.get(),
            max_blocks_per_seq, block_size, num_seqs, d_out.get(), out_features * sizeof(uint8_t), flusher
        );
    }

    {
        size_t total_weight_elements = static_cast<size_t>(vocab_size) * (out_features / 2);
        DeviceBuffer<uint8_t> d_weight(total_weight_elements, 0);

        run_embeddings_benchmark(
            static_cast<int32_t>(DataType::FP4), "FP4", total_tokens, out_features, vocab_size,
            d_weight.get(), d_weight_scales.get(), d_tokens.get(), d_seq_offsets.get(), d_block_table.get(), d_slot_mapping.get(),
            max_blocks_per_seq, block_size, num_seqs, d_out.get(), out_features / 2, flusher
        );
    }
}

REGISTER_BENCHMARK("embeddings", run_varlen_embeddings_benchmarks);
