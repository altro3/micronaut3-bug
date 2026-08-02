#include <doctest.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include "data_types.h"
#include "input_stage/fused_rmsnorm_forward.cuh"

struct RMSNormContext {
    const int32_t total_tokens = 4;
    const int32_t hidden_size = 128;
    const int32_t threads_per_block = 64;
    const float epsilon = 1e-5f;

    std::vector<float> h_flat_input;
    std::vector<float> h_flat_gamma;

    __nv_bfloat16 *d_out = nullptr;
    __nv_bfloat16 *d_gamma = nullptr;

    RMSNormContext() {
        h_flat_input.resize(total_tokens * hidden_size);
        h_flat_gamma.resize(hidden_size);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist_in(-1.5f, 1.5f);
        std::uniform_real_distribution<float> dist_g(0.5f, 1.5f);

        for (int32_t i = 0; i < total_tokens * hidden_size; ++i) h_flat_input[i] = dist_in(gen);
        for (int32_t i = 0; i < hidden_size; ++i) h_flat_gamma[i] = dist_g(gen);

        REQUIRE(cudaMalloc(&d_out, total_tokens * hidden_size * sizeof(__nv_bfloat16)) == cudaSuccess);
        REQUIRE(cudaMalloc(&d_gamma, hidden_size * sizeof(__nv_bfloat16)) == cudaSuccess);

        std::vector<__nv_bfloat16> h_gamma_bf16(hidden_size);
        for (int32_t i = 0; i < hidden_size; ++i) h_gamma_bf16[i] = __float2bfloat16(h_flat_gamma[i]);
        REQUIRE(cudaMemcpy(d_gamma, h_gamma_bf16.data(), hidden_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice) == cudaSuccess);
    }

    ~RMSNormContext() {
        cudaFree(d_out);
        cudaFree(d_gamma);
    }
};

TEST_CASE("FusedRMSNormTest - TestBF16") {

    printf("FusedRMSNormTest !!!!!!!!!!");

    RMSNormContext ctx;

    std::vector<__nv_bfloat16> h_input_bf16(ctx.total_tokens * ctx.hidden_size);
    for (int32_t i = 0; i < ctx.total_tokens * ctx.hidden_size; ++i) {
        h_input_bf16[i] = __float2bfloat16(ctx.h_flat_input[i]);
    }

    void *d_input = nullptr;
    REQUIRE(cudaMalloc(&d_input, ctx.total_tokens * ctx.hidden_size * sizeof(__nv_bfloat16)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_input, h_input_bf16.data(), ctx.total_tokens * ctx.hidden_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice) == cudaSuccess);

    launch_fused_rmsnorm_forward(
        ctx.d_out, d_input, ctx.d_gamma, nullptr, ctx.epsilon, ctx.total_tokens, ctx.hidden_size,
        static_cast<int32_t>(DataType::BF16), ctx.threads_per_block, nullptr
    );
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<__nv_bfloat16> h_out_res(ctx.total_tokens * ctx.hidden_size);
    REQUIRE(cudaMemcpy(h_out_res.data(), ctx.d_out, ctx.total_tokens * ctx.hidden_size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost) == cudaSuccess);

    for (int32_t t = 0; t < ctx.total_tokens; ++t) {
        float sum_sq = 0.0f;
        for (int32_t h = 0; h < ctx.hidden_size; ++h) {
            const float val = ctx.h_flat_input[t * ctx.hidden_size + h];
            sum_sq += val * val;
        }
        const float inv_rms = 1.0f / std::sqrt(sum_sq / ctx.hidden_size + ctx.epsilon);

        for (int32_t h = 0; h < ctx.hidden_size; ++h) {
            const float expected = ctx.h_flat_input[t * ctx.hidden_size + h] * inv_rms * ctx.h_flat_gamma[h];
            const float actual = __bfloat162float(h_out_res[t * ctx.hidden_size + h]);

            CHECK(actual == doctest::Approx(expected).epsilon(0.03));
        }
    }
    cudaFree(d_input);
}

TEST_CASE("FusedRMSNormTest - TestFP8") {
    RMSNormContext ctx;

    std::vector<__nv_fp8_e4m3> h_input_fp8(ctx.total_tokens * ctx.hidden_size);
    std::vector<float> h_quant_input(ctx.total_tokens * ctx.hidden_size);
    for (int32_t i = 0; i < ctx.total_tokens * ctx.hidden_size; ++i) {
        h_input_fp8[i] = __nv_fp8_e4m3(ctx.h_flat_input[i]);
        h_quant_input[i] = static_cast<float>(h_input_fp8[i]);
    }

    void *d_input = nullptr;
    REQUIRE(cudaMalloc(&d_input, ctx.total_tokens * ctx.hidden_size * sizeof(__nv_fp8_e4m3)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_input, h_input_fp8.data(), ctx.total_tokens * ctx.hidden_size * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice) == cudaSuccess);

    launch_fused_rmsnorm_forward(
        ctx.d_out, d_input, ctx.d_gamma, nullptr, ctx.epsilon, ctx.total_tokens, ctx.hidden_size,
        static_cast<int32_t>(DataType::FP8), ctx.threads_per_block, nullptr
    );
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<__nv_bfloat16> h_out_res(ctx.total_tokens * ctx.hidden_size);
    REQUIRE(cudaMemcpy(h_out_res.data(), ctx.d_out, ctx.total_tokens * ctx.hidden_size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost) == cudaSuccess);

    for (int32_t t = 0; t < ctx.total_tokens; ++t) {
        float sum_sq = 0.0f;
        for (int32_t h = 0; h < ctx.hidden_size; ++h) {
            const float val = h_quant_input[t * ctx.hidden_size + h];
            sum_sq += val * val;
        }
        const float inv_rms = 1.0f / std::sqrt(sum_sq / ctx.hidden_size + ctx.epsilon);

        for (int32_t h = 0; h < ctx.hidden_size; ++h) {
            const float expected = h_quant_input[t * ctx.hidden_size + h] * inv_rms * ctx.h_flat_gamma[h];
            const float actual = __bfloat162float(h_out_res[t * ctx.hidden_size + h]);

            CHECK(actual == doctest::Approx(expected).epsilon(0.02));
        }
    }
    cudaFree(d_input);
}

TEST_CASE("FusedRMSNormTest - TestFP4") {
    RMSNormContext ctx;

    const int32_t u32_per_row = ctx.hidden_size / 8;
    const int32_t scales_per_row = ctx.hidden_size / 32;

    std::vector<uint32_t> h_input_fp4(ctx.total_tokens * u32_per_row);
    const std::vector<float> h_gamma_scales(scales_per_row, 1.25f);
    std::vector<float> h_unpacked_input(ctx.total_tokens * ctx.hidden_size);

    for (int32_t t = 0; t < ctx.total_tokens; ++t) {
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
                h_unpacked_input[t * ctx.hidden_size + out_base + b * 2 + 0] = __half2float(h2.x);
                h_unpacked_input[t * ctx.hidden_size + out_base + b * 2 + 1] = __half2float(h2.y);
            }
        }
    }

    void *d_input = nullptr;
    float *d_gamma_scales = nullptr;
    REQUIRE(cudaMalloc(&d_input, ctx.total_tokens * u32_per_row * sizeof(uint32_t)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_gamma_scales, scales_per_row * sizeof(float)) == cudaSuccess);

    REQUIRE(cudaMemcpy(d_input, h_input_fp4.data(), ctx.total_tokens * u32_per_row * sizeof(uint32_t), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_gamma_scales, h_gamma_scales.data(), scales_per_row * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);

    launch_fused_rmsnorm_forward(
        ctx.d_out, d_input, ctx.d_gamma, d_gamma_scales, ctx.epsilon, ctx.total_tokens, ctx.hidden_size,
        static_cast<int32_t>(DataType::FP4), ctx.threads_per_block, nullptr
    );
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<__nv_bfloat16> h_out_res(ctx.total_tokens * ctx.hidden_size);
    REQUIRE(cudaMemcpy(h_out_res.data(), ctx.d_out, ctx.total_tokens * ctx.hidden_size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost) == cudaSuccess);

    for (int32_t t = 0; t < ctx.total_tokens; ++t) {
        float sum_sq = 0.0f;
        for (int32_t h = 0; h < ctx.hidden_size; ++h) {
            const float val = h_unpacked_input[t * ctx.hidden_size + h];
            sum_sq += val * val;
        }
        const float inv_rms = 1.0f / std::sqrt(sum_sq / ctx.hidden_size + ctx.epsilon);

        for (int32_t h = 0; h < ctx.hidden_size; ++h) {
            const int32_t scale_group = h / 32;
            const float current_scale = h_gamma_scales[scale_group];

            const float expected = h_unpacked_input[t * ctx.hidden_size + h] * current_scale * inv_rms * ctx.h_flat_gamma[h];
            const float actual = __bfloat162float(h_out_res[t * ctx.hidden_size + h]);

            CHECK(actual == doctest::Approx(expected).epsilon(0.03));
        }
    }
    cudaFree(d_input);
    cudaFree(d_gamma_scales);
}
