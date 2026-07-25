#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <vector>
#include <cmath>
#include <random>
#include <cstdint>
#include "data_types.h"

extern "C" {
void launch_fused_rmsnorm_forward(
    void *out,
    const void *input,
    const void *gamma,
    const float *gamma_scales,
    float epsilon,
    int32_t total_tokens,
    int32_t hidden_size,
    int32_t data_type,
    int32_t threads_per_block,
    void *stream_ptr
);
}

class FusedRMSNormTest : public testing::Test {
protected:
    const int32_t total_tokens = 4;
    const int32_t hidden_size = 128;
    const int32_t threads_per_block = 64;
    const float epsilon = 1e-5f;

    std::vector<float> h_flat_input;
    std::vector<float> h_flat_gamma;

    __nv_bfloat16 *d_out = nullptr;
    __nv_bfloat16 *d_gamma = nullptr;

    void SetUp() override {
        h_flat_input.resize(total_tokens * hidden_size);
        h_flat_gamma.resize(hidden_size);

        std::mt19937 gen(42);
        std::uniform_real_distribution dist_in(-1.5f, 1.5f);
        std::uniform_real_distribution dist_g(0.5f, 1.5f);

        for (int32_t i = 0; i < total_tokens * hidden_size; ++i) h_flat_input[i] = dist_in(gen);
        for (int32_t i = 0; i < hidden_size; ++i) h_flat_gamma[i] = dist_g(gen);

        ASSERT_EQ(cudaMalloc(&d_out, total_tokens * hidden_size * sizeof(__nv_bfloat16)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_gamma, hidden_size * sizeof(__nv_bfloat16)), cudaSuccess);

        std::vector<__nv_bfloat16> h_gamma_bf16(hidden_size);
        for (int32_t i = 0; i < hidden_size; ++i) h_gamma_bf16[i] = __float2bfloat16(h_flat_gamma[i]);
        ASSERT_EQ(cudaMemcpy(d_gamma, h_gamma_bf16.data(), hidden_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), cudaSuccess);
    }

    void TearDown() override {
        cudaFree(d_out);
        cudaFree(d_gamma);
    }
};

TEST_F(FusedRMSNormTest, TestBF16) {
    std::vector<__nv_bfloat16> h_input_bf16(total_tokens * hidden_size);
    for (int32_t i = 0; i < total_tokens * hidden_size; ++i) h_input_bf16[i] = __float2bfloat16(h_flat_input[i]);

    void *d_input;
    ASSERT_EQ(cudaMalloc(&d_input, total_tokens * hidden_size * sizeof(__nv_bfloat16)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_input, h_input_bf16.data(), total_tokens * hidden_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), cudaSuccess);

    launch_fused_rmsnorm_forward(d_out, d_input, d_gamma, nullptr, epsilon, total_tokens, hidden_size, static_cast<int32_t>(DataType::BF16), threads_per_block, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<__nv_bfloat16> h_out_res(total_tokens * hidden_size);
    ASSERT_EQ(cudaMemcpy(h_out_res.data(), d_out, total_tokens * hidden_size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), cudaSuccess);

    for (int32_t t = 0; t < total_tokens; ++t) {
        float sum_sq = 0.0f;
        for (int32_t h = 0; h < hidden_size; ++h) {
            const float val = h_flat_input[t * hidden_size + h];
            sum_sq += val * val;
        }
        const float inv_rms = 1.0f / std::sqrt(sum_sq / hidden_size + epsilon);

        for (int32_t h = 0; h < hidden_size; ++h) {
            const float expected = h_flat_input[t * hidden_size + h] * inv_rms * h_flat_gamma[h];
            const float actual = __bfloat162float(h_out_res[t * hidden_size + h]);
            EXPECT_NEAR(actual, expected, 3e-2f);
        }
    }
    cudaFree(d_input);
}

TEST_F(FusedRMSNormTest, TestFP8) {
    std::vector<__nv_fp8_e4m3> h_input_fp8(total_tokens * hidden_size);
    std::vector<float> h_quant_input(total_tokens * hidden_size);
    for (int32_t i = 0; i < total_tokens * hidden_size; ++i) {
        h_input_fp8[i] = static_cast<__nv_fp8_e4m3>(h_flat_input[i]);
        h_quant_input[i] = static_cast<float>(h_input_fp8[i]);
    }

    void *d_input;
    ASSERT_EQ(cudaMalloc(&d_input, total_tokens * hidden_size * sizeof(__nv_fp8_e4m3)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_input, h_input_fp8.data(), total_tokens * hidden_size * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice), cudaSuccess);

    launch_fused_rmsnorm_forward(d_out, d_input, d_gamma, nullptr, epsilon, total_tokens, hidden_size, static_cast<int32_t>(DataType::FP8), threads_per_block, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<__nv_bfloat16> h_out_res(total_tokens * hidden_size);
    ASSERT_EQ(cudaMemcpy(h_out_res.data(), d_out, total_tokens * hidden_size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), cudaSuccess);

    for (int32_t t = 0; t < total_tokens; ++t) {
        float sum_sq = 0.0f;
        for (int32_t h = 0; h < hidden_size; ++h) {
            const float val = h_quant_input[t * hidden_size + h];
            sum_sq += val * val;
        }
        const float inv_rms = 1.0f / std::sqrt(sum_sq / hidden_size + epsilon);

        for (int32_t h = 0; h < hidden_size; ++h) {
            const float expected = h_quant_input[t * hidden_size + h] * inv_rms * h_flat_gamma[h];
            const float actual = __bfloat162float(h_out_res[t * hidden_size + h]);
            EXPECT_NEAR(actual, expected, 2e-2f);
        }
    }
    cudaFree(d_input);
}

TEST_F(FusedRMSNormTest, TestFP4) {
    const int32_t u32_per_row = hidden_size / 8;
    const int32_t scales_per_row = hidden_size / 32;

    std::vector<uint32_t> h_input_fp4(total_tokens * u32_per_row);
    const std::vector h_gamma_scales(scales_per_row, 1.25f);
    std::vector<float> h_unpacked_input(total_tokens * hidden_size);

    for (int32_t t = 0; t < total_tokens; ++t) {
        for (int32_t i = 0; i < u32_per_row; ++i) {
            constexpr uint8_t b0 = 0x12;
            constexpr uint8_t b1 = 0x23;
            constexpr uint8_t b2 = 0x34;
            constexpr uint8_t b3 = 0x45;
            h_input_fp4[t * u32_per_row + i] = (b3 << 24) | (b2 << 16) | (b1 << 8) | b0;

            constexpr uint8_t bytes[4] = {b0, b1, b2, b3};
            const int32_t out_base = i * 8;
            for (int32_t b = 0; b < 4; ++b) {
                __half2_raw raw_h2 = __nv_cvt_fp4x2_to_halfraw2(bytes[b], __NV_E2M1);
                const __half2 h2 = *reinterpret_cast<__half2 *>(&raw_h2);
                h_unpacked_input[t * hidden_size + out_base + b * 2 + 0] = __half2float(h2.x);
                h_unpacked_input[t * hidden_size + out_base + b * 2 + 1] = __half2float(h2.y);
            }
        }
    }

    void *d_input;
    float *d_gamma_scales;
    ASSERT_EQ(cudaMalloc(&d_input, total_tokens * u32_per_row * sizeof(uint32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_gamma_scales, scales_per_row * sizeof(float)), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_input, h_input_fp4.data(), total_tokens * u32_per_row * sizeof(uint32_t), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_gamma_scales, h_gamma_scales.data(), scales_per_row * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    launch_fused_rmsnorm_forward(d_out, d_input, d_gamma, d_gamma_scales, epsilon, total_tokens, hidden_size, static_cast<int32_t>(DataType::FP4), threads_per_block, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<__nv_bfloat16> h_out_res(total_tokens * hidden_size);
    ASSERT_EQ(cudaMemcpy(h_out_res.data(), d_out, total_tokens * hidden_size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), cudaSuccess);

    for (int32_t t = 0; t < total_tokens; ++t) {
        float sum_sq = 0.0f;
        for (int32_t h = 0; h < hidden_size; ++h) {
            const float val = h_unpacked_input[t * hidden_size + h];
            sum_sq += val * val;
        }
        const float inv_rms = 1.0f / std::sqrt(sum_sq / hidden_size + epsilon);

        for (int32_t h = 0; h < hidden_size; ++h) {
            const int32_t scale_group = h / 32;
            const float current_scale = h_gamma_scales[scale_group];

            const float expected = h_unpacked_input[t * hidden_size + h] * current_scale * inv_rms * h_flat_gamma[h];
            const float actual = __bfloat162float(h_out_res[t * hidden_size + h]);
            EXPECT_NEAR(actual, expected, 3e-2f);
        }
    }
    cudaFree(d_input);
    cudaFree(d_gamma_scales);
}
