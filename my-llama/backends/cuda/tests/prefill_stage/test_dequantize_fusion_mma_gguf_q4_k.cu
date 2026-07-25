#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <vector>
#include "data_types.h"
#include "prefill_stage/dequantize_fusion_mma_gguf_q4_k.cuh"

extern "C" void launch_fused_gemm_gguf_q4_k(
    void *output_activations, const void *input_activations, const void *quantized_weights,
    int32_t batch_size_or_tokens, int32_t hidden_units_out, int32_t hidden_units_in,
    int32_t data_type, void *stream_ptr
);

class GgufQ4KGemmTest : public ::testing::Test {
protected:
    const int32_t M = 64;
    const int32_t N = 64;
    const int32_t K = 256;

    std::vector<BlockQ4K> h_weights_quant;
    std::vector<float> h_unpacked_weights;

    void SetUp() override {
        const int32_t total_weight_elements = N * K;
        const int32_t total_blocks = total_weight_elements / 256;
        h_weights_quant.resize(total_blocks);
        h_unpacked_weights.resize(total_weight_elements);

        for (int32_t b = 0; b < total_blocks; ++b) {
            h_weights_quant[b].d = __float2bfloat16(0.01f);
            h_weights_quant[b].dmin = __float2bfloat16(0.002f);

            for (int32_t s = 0; s < 12; ++s) {
                h_weights_quant[b].scales[s] = 4;
            }
            for (int32_t q = 0; q < 128; ++q) {
                h_weights_quant[b].qs[q] = 0x22;
            }
        }

        for (int32_t n = 0; n < N; ++n) {
            for (int32_t k = 0; k < K; ++k) {
                const int32_t idx = n * K + k;
                const int32_t b_idx = n * (K / 256) + (k / 256);
                const int32_t elem_in_b = k % 256;

                const int32_t sub_block_idx = elem_in_b / 32;
                const int32_t elem_idx = elem_in_b % 32;

                const int32_t bit_offset_sc = sub_block_idx * 6;
                const int32_t byte_offset_sc = bit_offset_sc / 8;
                const int32_t bit_shift_sc = bit_offset_sc % 8;
                uint32_t val_sc = h_weights_quant[b_idx].scales[byte_offset_sc] | (h_weights_quant[b_idx].scales[byte_offset_sc + 1] << 8);
                if (byte_offset_sc + 2 < 12) {
                    val_sc |= (h_weights_quant[b_idx].scales[byte_offset_sc + 2] << 16);
                }
                const uint8_t sc = (val_sc >> bit_shift_sc) & 0x3F;

                const int32_t bit_offset_min = (sub_block_idx + 8) * 6;
                const int32_t byte_offset_min = bit_offset_min / 8;
                const int32_t bit_shift_min = bit_offset_min % 8;
                uint32_t val_min = h_weights_quant[b_idx].scales[byte_offset_min] | (h_weights_quant[b_idx].scales[byte_offset_min + 1] << 8);
                if (byte_offset_min + 2 < 12) {
                    val_min |= (h_weights_quant[b_idx].scales[byte_offset_min + 2] << 16);
                }
                const uint8_t min_sc = (val_min >> bit_shift_min) & 0x3F;

                const uint8_t qs_byte = h_weights_quant[b_idx].qs[sub_block_idx * 16 + (elem_idx % 16)];
                const uint8_t raw_q = (elem_idx < 16) ? (qs_byte & 0x0F) : (qs_byte >> 4);

                h_unpacked_weights[idx] = 0.01f * static_cast<float>(sc) * static_cast<float>(raw_q) - 0.002f * static_cast<float>(min_sc);
            }
        }
    }
};

TEST_F(GgufQ4KGemmTest, TestBF16) {
    const std::vector h_input(M * K, __float2bfloat16(0.1f));
    std::vector<__nv_bfloat16> h_output(M * N);

    void *d_out, *d_in, *d_w;
    ASSERT_EQ(cudaMalloc(&d_out, M * N * sizeof(__nv_bfloat16)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_in, M * K * sizeof(__nv_bfloat16)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_w, h_weights_quant.size() * sizeof(BlockQ4K)), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_in, h_input.data(), h_input.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_w, h_weights_quant.data(), h_weights_quant.size() * sizeof(BlockQ4K), cudaMemcpyHostToDevice), cudaSuccess);

    launch_fused_gemm_gguf_q4_k(d_out, d_in, d_w, M, N, K, static_cast<int32_t>(DataType::BF16), nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(h_output.data(), d_out, h_output.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), cudaSuccess);

    for (int32_t m = 0; m < M; ++m) {
        for (int32_t n = 0; n < N; ++n) {
            float expected = 0.0f;
            for (int32_t k = 0; k < K; ++k) {
                expected += 0.1f * h_unpacked_weights[n * K + k];
            }
            const float actual = __bfloat162float(h_output[m * N + n]);
            EXPECT_NEAR(actual, expected, 1.0f);
        }
    }
    cudaFree(d_out);
    cudaFree(d_in);
    cudaFree(d_w);
}

TEST_F(GgufQ4KGemmTest, TestFP8) {
    std::vector<__nv_fp8_e4m3> h_input(M * K);
    for (auto &val: h_input) val = static_cast<__nv_fp8_e4m3>(0.25f);
    std::vector<__nv_fp8_e4m3> h_output(M * N);

    void *d_out, *d_in, *d_w;
    ASSERT_EQ(cudaMalloc(&d_out, M * N * sizeof(__nv_fp8_e4m3)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_in, M * K * sizeof(__nv_fp8_e4m3)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_w, h_weights_quant.size() * sizeof(BlockQ4K)), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_in, h_input.data(), h_input.size() * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_w, h_weights_quant.data(), h_weights_quant.size() * sizeof(BlockQ4K), cudaMemcpyHostToDevice), cudaSuccess);

    launch_fused_gemm_gguf_q4_k(d_out, d_in, d_w, M, N, K, static_cast<int32_t>(DataType::FP8), nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(h_output.data(), d_out, h_output.size() * sizeof(__nv_fp8_e4m3), cudaMemcpyDeviceToHost), cudaSuccess);

    const float in_val = static_cast<float>(static_cast<__nv_fp8_e4m3>(0.25f));
    for (int32_t m = 0; m < M; ++m) {
        for (int32_t n = 0; n < N; ++n) {
            float expected = 0.0f;
            for (int32_t k = 0; k < K; ++k) {
                expected += in_val * h_unpacked_weights[n * K + k];
            }
            const float actual = static_cast<float>(h_output[m * N + n]);
            EXPECT_NEAR(actual, expected, 1.0f);
        }
    }
    cudaFree(d_out);
    cudaFree(d_in);
    cudaFree(d_w);
}

TEST_F(GgufQ4KGemmTest, TestFP4) {
    const std::vector<uint8_t> h_input(M * K / 2, 0x11);
    std::vector<uint8_t> h_output(M * N / 2, 0xFF);

    void *d_out, *d_in, *d_w;
    ASSERT_EQ(cudaMalloc(&d_out, (M * N / 2) * sizeof(uint8_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_in, (M * K / 2) * sizeof(uint8_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_w, h_weights_quant.size() * sizeof(BlockQ4K)), cudaSuccess);

    ASSERT_EQ(cudaMemset(d_out, 0, (M * N / 2) * sizeof(uint8_t)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_in, h_input.data(), h_input.size() * sizeof(uint8_t), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_w, h_weights_quant.data(), h_weights_quant.size() * sizeof(BlockQ4K), cudaMemcpyHostToDevice), cudaSuccess);

    launch_fused_gemm_gguf_q4_k(d_out, d_in, d_w, M, N, K, static_cast<int32_t>(DataType::FP4), nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(h_output.data(), d_out, h_output.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost), cudaSuccess);

    __half2_raw raw_h2_in = __nv_cvt_fp4x2_to_halfraw2(0x11, __NV_E2M1);
    const float in_val = __half2float(reinterpret_cast<__half2 *>(&raw_h2_in)->x);

    bool has_error = false;
    for (int32_t m = 0; m < M && !has_error; ++m) {
        for (int32_t n = 0; n < N && !has_error; ++n) {
            const int32_t global_element_idx = m * N + n;
            const int32_t global_u32_idx = global_element_idx / 8;
            const int32_t shift = global_element_idx % 8 * 4;
            const auto h_output_u32 = reinterpret_cast<const uint32_t *>(h_output.data());
            const uint8_t nibble = (h_output_u32[global_u32_idx] >> shift) & 0x0F;
            const uint8_t aligned_fp4x2 = (nibble << 4) | nibble;
            __half2_raw r0 = __nv_cvt_fp4x2_to_halfraw2(aligned_fp4x2, __NV_E2M1);
            const float actual = __half2float(reinterpret_cast<__half2 *>(&r0)->x) * 2.0f;

            float expected_accum = 0.0f;
            for (int32_t k = 0; k < K; ++k) {
                expected_accum += in_val * h_unpacked_weights[n * K + k];
            }

            if (abs(actual - expected_accum) > 2.5f) {
                printf("[TEST ERROR] First mismatch at M=%d, N=%d | Actual: %f, Expected: %f\n", m, n, actual, expected_accum);
                has_error = true;
            }
        }
    }
    EXPECT_FALSE(has_error);
    cudaFree(d_out);
    cudaFree(d_in);
    cudaFree(d_w);
}
