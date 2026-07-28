#include <doctest/doctest.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>
#include <iostream>
#include "prefill_stage/dequantize_fusion_mma_gguf_q4_k_blackwell.cuh"

struct BlackwellGgufNativeTestContext {
    const int32_t M = 1024;
    const int32_t N = 4096;
    const int32_t K = 4096;

    std::vector<BlockQ4K> h_weights_quant;
    std::vector<float> h_unpacked_weights;

    BlackwellGgufNativeTestContext() {
        const int32_t total_weight_elements = N * K;
        const int32_t total_blocks = total_weight_elements / 256;
        h_weights_quant.resize(total_blocks);
        h_unpacked_weights.resize(total_weight_elements);

        for (int32_t b = 0; b < total_blocks; ++b) {
            h_weights_quant[b].d = __float2bfloat16(0.02f);
            h_weights_quant[b].dmin = __float2bfloat16(0.005f);

            for (int32_t s = 0; s < 12; ++s) {
                h_weights_quant[b].scales[s] = 8;
            }
            for (int32_t q = 0; q < 128; ++q) {
                h_weights_quant[b].qs[q] = 0x33;
            }
        }

        for (int32_t k = 0; k < K; ++k) {
            for (int32_t n = 0; n < N; ++n) {
                const int32_t idx = n * K + k;

                const int32_t total_blocks_k = K / 256;
                const int32_t block_k_idx = k / 256;
                const int32_t local_k_idx = k % 256;
                const int32_t b_idx = n * total_blocks_k + block_k_idx;

                const int32_t sub_block_idx = local_k_idx / 32;
                const int32_t elem_idx = local_k_idx % 32;

                const uint8_t sc_byte = h_weights_quant[b_idx].scales[sub_block_idx * 2 + elem_idx / 16];
                float scale = elem_idx % 16 < 8 ? sc_byte & 0x0F : sc_byte >> 4;

                const uint8_t q_byte = h_weights_quant[b_idx].qs[(sub_block_idx * 32 + elem_idx) / 2];
                uint8_t q_raw = elem_idx % 2 == 0 ? q_byte & 0x0F : q_byte >> 4;

                float d_val = __bfloat162float(h_weights_quant[b_idx].d);
                float dmin_val = __bfloat162float(h_weights_quant[b_idx].dmin);

                float raw_weight = d_val * scale * q_raw - dmin_val;
                h_unpacked_weights[idx] = raw_weight;
            }
        }
    }
};

TEST_CASE("GgufBlackwellPrefillTest - AccuracyVerification") {
    BlackwellGgufNativeTestContext ctx;

    std::vector<__nv_bfloat16> h_input(ctx.M * ctx.K, __float2bfloat16(0.5f));
    std::vector<__nv_bfloat16> h_output(ctx.M * ctx.N, __float2bfloat16(0.0f));

    void *d_out = nullptr;
    void *d_in = nullptr;
    void *d_w = nullptr;

    REQUIRE(cudaMalloc(&d_out, ctx.M * ctx.N * sizeof(__nv_bfloat16)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_in, ctx.M * ctx.K * sizeof(__nv_bfloat16)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_w, ctx.h_weights_quant.size() * sizeof(BlockQ4K)) == cudaSuccess);

    REQUIRE(cudaMemcpy(d_in, h_input.data(), h_input.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_w, ctx.h_weights_quant.data(), ctx.h_weights_quant.size() * sizeof(BlockQ4K), cudaMemcpyHostToDevice) == cudaSuccess);

    launch_fused_gemm_gguf_blackwell_fp4_native(
        d_out,
        d_in,
        d_w,
        ctx.M,
        ctx.N,
        ctx.K,
        nullptr
    );

    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    REQUIRE(cudaMemcpy(h_output.data(), d_out, h_output.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost) == cudaSuccess);

    bool has_error = false;
    for (int32_t m = 0; m < ctx.M && !has_error; ++m) {
        for (int32_t n = 0; n < ctx.N && !has_error; ++n) {
            float expected = 0.0f;
            for (int32_t k = 0; k < ctx.K; ++k) {
                expected += __bfloat162float(h_input[m * ctx.K + k]) * ctx.h_unpacked_weights[k * ctx.N + n];
            }

            const float actual = __bfloat162float(h_output[m * ctx.N + n]);

            if (std::abs(actual - expected) > 1.5f) {
                printf("[TEST ACCURACY ERROR] Mismatch at token M=%d, unit N=%d | Actual kernel out: %f, Expected host math: %f\n", m, n, actual, expected);
                has_error = true;
            }
        }
    }

    CHECK_FALSE(has_error);

    cudaFree(d_out);
    cudaFree(d_in);
    cudaFree(d_w);
}
