#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <vector>
#include <cstdint>
#include "data_types.h"

extern "C" {
void launch_varlen_embeddings(
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
}

class VarlenEmbeddingsTest : public testing::Test {
protected:
    const int32_t num_seqs = 3;
    const int32_t block_size = 4;
    const int32_t max_blocks_per_seq = 2;
    const int32_t out_features = 64;
    const int32_t vocab_size = 10;
    const int32_t threads_per_block = 128;
    int32_t total_tokens = 0;

    std::vector<int32_t> h_seq_offsets = {0, 3, 7, 8};
    std::vector<uint32_t> h_tokens = {0, 1, 99, 2, 3, 4, 5, 0};
    std::vector<int32_t> h_block_table = {
        10, -1,
        12, 15,
        20, -1
    };
    std::vector<int32_t> h_expected_slots;

    __nv_bfloat16 *d_out = nullptr;
    uint32_t *d_tokens = nullptr;
    int32_t *d_seq_offsets = nullptr;
    int32_t *d_block_table = nullptr;
    int32_t *d_slot_mapping = nullptr;

    static int32_t host_find_sequence_index(const int32_t *offsets, int32_t num_seqs, int32_t token_idx) {
        int32_t low = 0;
        int32_t high = num_seqs - 1;
        int32_t seq_idx = 0;
        while (low <= high) {
            int32_t mid = low + (high - low) / 2;
            if (offsets[mid] <= token_idx) {
                seq_idx = mid;
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        return seq_idx;
    }

    void SetUp() override {
        total_tokens = h_seq_offsets.back();
        h_expected_slots.resize(total_tokens);

        for (int32_t i = 0; i < total_tokens; ++i) {
            int32_t seq_idx = host_find_sequence_index(h_seq_offsets.data(), num_seqs, i);
            int32_t start_tok_idx = h_seq_offsets[seq_idx];
            int32_t token_local_idx = i - start_tok_idx;
            int32_t logical_block_idx = token_local_idx / block_size;
            int32_t block_offset = token_local_idx % block_size;
            int32_t physical_block_id = h_block_table[seq_idx * max_blocks_per_seq + logical_block_idx];
            h_expected_slots[i] = (physical_block_id != -1) ? (physical_block_id * block_size + block_offset) : -1;
        }

        ASSERT_EQ(cudaMalloc(&d_out, total_tokens * out_features * sizeof(__nv_bfloat16)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_tokens, total_tokens * sizeof(uint32_t)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_seq_offsets, h_seq_offsets.size() * sizeof(int32_t)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_block_table, h_block_table.size() * sizeof(int32_t)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_slot_mapping, total_tokens * sizeof(int32_t)), cudaSuccess);

        ASSERT_EQ(cudaMemcpy(d_tokens, h_tokens.data(), total_tokens * sizeof(uint32_t), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_seq_offsets, h_seq_offsets.data(), h_seq_offsets.size() * sizeof(int32_t), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_block_table, h_block_table.data(), h_block_table.size() * sizeof(int32_t), cudaMemcpyHostToDevice), cudaSuccess);
    }

    void TearDown() override {
        cudaFree(d_out);
        cudaFree(d_tokens);
        cudaFree(d_seq_offsets);
        cudaFree(d_block_table);
        cudaFree(d_slot_mapping);
    }
};

TEST_F(VarlenEmbeddingsTest, TestBF16) {
    std::vector<__nv_bfloat16> h_weight_bf16(vocab_size * out_features);
    for (int32_t i = 0; i < vocab_size * out_features; ++i) {
        h_weight_bf16[i] = __float2bfloat16(static_cast<float>(i % 5) * 0.25f + 0.1f);
    }

    void *d_weight_bf16;
    ASSERT_EQ(cudaMalloc(&d_weight_bf16, vocab_size * out_features * sizeof(__nv_bfloat16)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_weight_bf16, h_weight_bf16.data(), vocab_size * out_features * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_slot_mapping, -1, total_tokens * sizeof(int32_t)), cudaSuccess);

    launch_varlen_embeddings(
        d_out, d_weight_bf16, nullptr, d_tokens, d_seq_offsets, d_block_table, d_slot_mapping,
        max_blocks_per_seq, block_size, total_tokens, out_features, vocab_size, num_seqs,
        static_cast<int32_t>(DataType::BF16), threads_per_block, nullptr
    );
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<__nv_bfloat16> h_out_res(total_tokens * out_features);
    std::vector<int32_t> h_slots_res(total_tokens);
    ASSERT_EQ(cudaMemcpy(h_out_res.data(), d_out, total_tokens * out_features * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(h_slots_res.data(), d_slot_mapping, total_tokens * sizeof(int32_t), cudaMemcpyDeviceToHost), cudaSuccess);

    for (int32_t i = 0; i < total_tokens; ++i) {
        EXPECT_EQ(h_slots_res[i], h_expected_slots[i]);
        uint32_t tok = h_tokens[i];
        for (int32_t f = 0; f < out_features; ++f) {
            float act = __bfloat162float(h_out_res[i * out_features + f]);
            if (tok >= static_cast<uint32_t>(vocab_size)) {
                EXPECT_NEAR(act, 0.0f, 1e-5f);
            } else {
                float exp = __bfloat162float(h_weight_bf16[tok * out_features + f]);
                EXPECT_NEAR(act, exp, 1e-5f);
            }
        }
    }
    cudaFree(d_weight_bf16);
}

TEST_F(VarlenEmbeddingsTest, TestFP8) {
    std::vector<__nv_fp8_e4m3> h_weight_fp8(vocab_size * out_features);
    std::vector<float> h_ref_floats(vocab_size * out_features);
    for (int32_t i = 0; i < vocab_size * out_features; ++i) {
        float val = static_cast<float>(i % 4) * 0.5f + 0.25f;
        h_weight_fp8[i] = static_cast<__nv_fp8_e4m3>(val);
        h_ref_floats[i] = static_cast<float>(h_weight_fp8[i]);
    }

    void *d_weight_fp8;
    ASSERT_EQ(cudaMalloc(&d_weight_fp8, vocab_size * out_features * sizeof(__nv_fp8_e4m3)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_weight_fp8, h_weight_fp8.data(), vocab_size * out_features * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice), cudaSuccess);

    launch_varlen_embeddings(
        d_out, d_weight_fp8, nullptr, d_tokens, d_seq_offsets, d_block_table, nullptr,
        max_blocks_per_seq, block_size, total_tokens, out_features, vocab_size, num_seqs,
        static_cast<int32_t>(DataType::FP8), threads_per_block, nullptr
    );
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<__nv_bfloat16> h_out_res(total_tokens * out_features);
    ASSERT_EQ(cudaMemcpy(h_out_res.data(), d_out, total_tokens * out_features * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), cudaSuccess);

    for (int32_t i = 0; i < total_tokens; ++i) {
        uint32_t tok = h_tokens[i];
        for (int32_t f = 0; f < out_features; ++f) {
            float act = __bfloat162float(h_out_res[i * out_features + f]);
            if (tok >= static_cast<uint32_t>(vocab_size)) {
                EXPECT_NEAR(act, 0.0f, 1e-5f);
            } else {
                float exp = h_ref_floats[tok * out_features + f];
                EXPECT_NEAR(act, exp, 1e-2f);
            }
        }
    }
    cudaFree(d_weight_fp8);
}

TEST_F(VarlenEmbeddingsTest, TestFP4) {
    const int32_t u32_per_row = out_features / 8;
    const int32_t scales_per_row = out_features / 32;
    std::vector<uint32_t> h_weight_fp4(vocab_size * u32_per_row);
    std::vector<float> h_scales_fp4(vocab_size * scales_per_row);
    std::vector<float> h_ref_floats(vocab_size * out_features);

    for (int32_t v = 0; v < vocab_size; ++v) {
        for (int32_t s = 0; s < scales_per_row; ++s) {
            h_scales_fp4[v * scales_per_row + s] = 1.5f;
        }
        for (int32_t i = 0; i < u32_per_row; ++i) {
            uint8_t b0 = 0x11;
            uint8_t b1 = 0x22;
            uint8_t b2 = 0x33;
            uint8_t b3 = 0x44;
            uint32_t packed = (b3 << 24) | (b2 << 16) | (b1 << 8) | b0;
            h_weight_fp4[v * u32_per_row + i] = packed;

            int32_t group_idx = i / 4;
            float current_scale = h_scales_fp4[v * scales_per_row + group_idx];

            uint8_t bytes[4] = {b0, b1, b2, b3};
            for (int32_t b = 0; b < 4; ++b) {
                __half2_raw raw_h2 = __nv_cvt_fp4x2_to_halfraw2(bytes[b], __NV_E2M1);
                __half2 h2 = *reinterpret_cast<__half2 *>(&raw_h2);
                h_ref_floats[v * out_features + i * 8 + b * 2 + 0] = __half2float(h2.x) * current_scale;
                h_ref_floats[v * out_features + i * 8 + b * 2 + 1] = __half2float(h2.y) * current_scale;
            }
        }
    }

    void *d_weight_fp4;
    float *d_scales_fp4;
    ASSERT_EQ(cudaMalloc(&d_weight_fp4, h_weight_fp4.size() * sizeof(uint32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_scales_fp4, h_scales_fp4.size() * sizeof(float)), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_weight_fp4, h_weight_fp4.data(), h_weight_fp4.size() * sizeof(uint32_t), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_scales_fp4, h_scales_fp4.data(), h_scales_fp4.size() * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    launch_varlen_embeddings(
        d_out, d_weight_fp4, d_scales_fp4, d_tokens, d_seq_offsets, d_block_table, nullptr,
        max_blocks_per_seq, block_size, total_tokens, out_features, vocab_size, num_seqs,
        static_cast<int32_t>(DataType::FP4), threads_per_block, nullptr
    );
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<__nv_bfloat16> h_out_res(total_tokens * out_features);
    ASSERT_EQ(cudaMemcpy(h_out_res.data(), d_out, total_tokens * out_features * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), cudaSuccess);

    for (int32_t i = 0; i < total_tokens; ++i) {
        uint32_t tok = h_tokens[i];
        for (int32_t f = 0; f < out_features; ++f) {
            float act = __bfloat162float(h_out_res[i * out_features + f]);
            if (tok >= static_cast<uint32_t>(vocab_size)) {
                EXPECT_NEAR(act, 0.0f, 1e-5f);
            } else {
                float exp = h_ref_floats[tok * out_features + f];
                EXPECT_NEAR(act, exp, 1e-3f);
            }
        }
    }
    cudaFree(d_weight_fp4);
    cudaFree(d_scales_fp4);
}
