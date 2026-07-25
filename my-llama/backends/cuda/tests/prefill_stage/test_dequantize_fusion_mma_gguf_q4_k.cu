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
            h_weights_quant[b].d = __float2bfloat16(0.5f);
            h_weights_quant[b].dmin = __float2bfloat16(0.1f);

            for (int32_t s = 0; s < 12; ++s) {
                h_weights_quant[b].scales[s] = 0x22;
            }
            for (int32_t q = 0; q < 128; ++q) {
                h_weights_quant[b].qs[q] = 0x33;
            }
        }

        for (int32_t n = 0; n < N; ++n) {
            for (int32_t k = 0; k < K; ++k) {
                const int32_t idx = n * K + k;
                const int32_t b_idx = idx / 256;
                const int32_t elem_in_b = idx & 255;
                const int32_t j = elem_in_b >> 6;
                const int32_t il = (elem_in_b >> 4) & 3;
                const int32_t pair_idx = elem_in_b & 15;
                const int32_t j_mod = j & 1;

                const uint8_t s_low = h_weights_quant[b_idx].scales[j_mod * 4 + il];
                const uint8_t sc = s_low & 63;
                const uint8_t s_high = h_weights_quant[b_idx].scales[8 + j_mod * 2 + (il >> 1)];
                const uint8_t min_sc = s_high & 63;

                const float d_super = 0.5f * static_cast<float>(sc);
                const float m_super = 0.1f * static_cast<float>(min_sc);

                const int32_t byte_idx = (j << 5) + (il << 2) + (pair_idx >> 1);
                const uint8_t packed_byte = h_weights_quant[b_idx].qs[byte_idx];
                const uint8_t raw_q = (pair_idx & 1) == 0 ? packed_byte & 0x0F : packed_byte >> 4;

                h_unpacked_weights[idx] = d_super * static_cast<float>(raw_q) - m_super;
            }
        }
    }
};

TEST_F(GgufQ4KGemmTest, TestBF16) {
    const std::vector<__nv_bfloat16> h_input(M * K, __float2bfloat16(0.25f));
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
                expected += 0.25f * h_unpacked_weights[n * K + k];
            }
            const float actual = __bfloat162float(h_output[m * N + n]);
            EXPECT_NEAR(actual, expected, 5e-1f);
        }
    }
    cudaFree(d_out);
    cudaFree(d_in);
    cudaFree(d_w);
}

TEST_F(GgufQ4KGemmTest, TestFP8) {
    std::vector<__nv_fp8_e4m3> h_input(M * K);
    for (auto &val: h_input) val = static_cast<__nv_fp8_e4m3>(0.5f);
    std::vector<__nv_bfloat16> h_output(M * N);

    void *d_out, *d_in, *d_w;
    ASSERT_EQ(cudaMalloc(&d_out, M * N * sizeof(__nv_bfloat16)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_in, M * K * sizeof(__nv_fp8_e4m3)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_w, h_weights_quant.size() * sizeof(BlockQ4K)), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_in, h_input.data(), h_input.size() * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_w, h_weights_quant.data(), h_weights_quant.size() * sizeof(BlockQ4K), cudaMemcpyHostToDevice), cudaSuccess);

    launch_fused_gemm_gguf_q4_k(d_out, d_in, d_w, M, N, K, static_cast<int32_t>(DataType::FP8), nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(h_output.data(), d_out, h_output.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), cudaSuccess);

    const float in_val = static_cast<float>(static_cast<__nv_fp8_e4m3>(0.5f));
    for (int32_t m = 0; m < M; ++m) {
        for (int32_t n = 0; n < N; ++n) {
            float expected = 0.0f;
            for (int32_t k = 0; k < K; ++k) {
                expected += in_val * h_unpacked_weights[n * K + k];
            }
            const float actual = __bfloat162float(h_output[m * N + n]);
            EXPECT_NEAR(actual, expected, 5e-1f);
        }
    }
    cudaFree(d_out);
    cudaFree(d_in);
    cudaFree(d_w);
}

TEST_F(GgufQ4KGemmTest, TestFP4) {
    const std::vector<uint8_t> h_input(M * K / 2, 0x11);
    std::vector<uint8_t> h_output(M * N / 2);

    void *d_out, *d_in, *d_w;
    ASSERT_EQ(cudaMalloc(&d_out, M * N / 2 * sizeof(uint8_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_in, M * K / 2 * sizeof(uint8_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_w, h_weights_quant.size() * sizeof(BlockQ4K)), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_in, h_input.data(), h_input.size() * sizeof(uint8_t), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_w, h_weights_quant.data(), h_weights_quant.size() * sizeof(BlockQ4K), cudaMemcpyHostToDevice), cudaSuccess);

    launch_fused_gemm_gguf_q4_k(d_out, d_in, d_w, M, N, K, static_cast<int32_t>(DataType::FP4), nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(h_output.data(), d_out, h_output.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost), cudaSuccess);

    __half2_raw raw_h2 = __nv_cvt_fp4x2_to_halfraw2(0x11, __NV_E2M1);
    const float in_val = __half2float((*reinterpret_cast<__half2 *>(&raw_h2)).x);

    std::vector<__nv_bfloat16> h_unpacked_out(M * N);
    for (size_t i = 0; i < h_output.size(); ++i) {
        const uint8_t b = h_output[i];
        __half2_raw r0 = __nv_cvt_fp4x2_to_halfraw2(b, __NV_E2M1);
        const __half2 h2 = *reinterpret_cast<__half2 *>(&r0);
        h_unpacked_out[i * 2 + 0] = __float2bfloat16(__half2float(h2.x));
        h_unpacked_out[i * 2 + 1] = __float2bfloat16(__half2float(h2.y));
    }

    for (int32_t m = 0; m < M; ++m) {
        for (int32_t n = 0; n < N; ++n) {
            float expected = 0.0f;
            for (int32_t k = 0; k < K; ++k) {
                expected += in_val * h_unpacked_weights[n * K + k];
            }
            const float actual = __bfloat162float(h_unpacked_out[m * N + n]);
            EXPECT_NEAR(actual, expected, 1.5f);
        }
    }
    cudaFree(d_out);
    cudaFree(d_in);
    cudaFree(d_w);
}
