#include <doctest/doctest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp4.h>

#include "data_types.h"

struct __nv_fp8_e4m3;
struct __nv_bfloat16;

extern "C" void launch_fused_multimodal_projection(
    const void **host_ptr_A,
    const void **host_ptr_B,
    void **host_ptr_D,
    const float *bias,
    const float *weight_scales,
    const int32_t *host_problem_shapes,
    int32_t num_segments,
    int32_t vision_hidden_size,
    int32_t text_hidden_size,
    int32_t tp_rank,
    int32_t tp_size,
    int32_t data_type,
    void *stream_ptr
);

struct MultimodalProjectionContext {
    const int32_t num_segments = 2;
    const int32_t vision_hidden_size = 64;
    const int32_t text_hidden_size = 128;
    const int32_t tp_rank = 0;
    const int32_t tp_size = 2;
    int32_t local_output_dim = 0;

    std::vector<int32_t> h_shapes = {
        64, 0, 0,
        64, 0, 0
    };

    std::vector<float> h_bias;
    std::vector<float> h_scales = {1.5f, 2.0f};

    float *d_bias = nullptr;
    float *d_scales = nullptr;

    MultimodalProjectionContext() {
        local_output_dim = text_hidden_size / tp_size;
        h_bias.assign(text_hidden_size, 0.25f);

        REQUIRE(cudaMalloc(&d_bias, text_hidden_size * sizeof(float)) == cudaSuccess);
        REQUIRE(cudaMalloc(&d_scales, num_segments * sizeof(float)) == cudaSuccess);

        REQUIRE(cudaMemcpy(d_bias, h_bias.data(), text_hidden_size * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaMemcpy(d_scales, h_scales.data(), num_segments * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    }

    ~MultimodalProjectionContext() {
        cudaFree(d_bias);
        cudaFree(d_scales);
    }
};

TEST_CASE("MultimodalProjectionTest - TestBF16") {
    MultimodalProjectionContext ctx;

    std::vector<const void *> host_A(ctx.num_segments);
    std::vector<const void *> host_B(ctx.num_segments);
    std::vector<void *> host_D(ctx.num_segments);

    std::vector<std::vector<__nv_bfloat16> > h_inputs(ctx.num_segments);
    std::vector<std::vector<__nv_bfloat16> > h_weights(ctx.num_segments);
    std::vector<std::vector<__nv_bfloat16> > h_outputs(ctx.num_segments);

    for (int32_t s = 0; s < ctx.num_segments; ++s) {
        const int32_t m = ctx.h_shapes[s * 3 + 0];
        h_inputs[s].assign(m * ctx.vision_hidden_size, __float2bfloat16(0.1f));
        h_weights[s].assign(ctx.local_output_dim * ctx.vision_hidden_size, __float2bfloat16(0.2f));
        h_outputs[s].resize(m * ctx.local_output_dim);

        REQUIRE(cudaMalloc(const_cast<void **>(&host_A[s]), m * ctx.vision_hidden_size * sizeof(__nv_bfloat16)) == cudaSuccess);
        REQUIRE(cudaMalloc(const_cast<void **>(&host_B[s]), ctx.local_output_dim * ctx.vision_hidden_size * sizeof(__nv_bfloat16)) == cudaSuccess);
        REQUIRE(cudaMalloc(&host_D[s], m * ctx.local_output_dim * sizeof(__nv_bfloat16)) == cudaSuccess);

        REQUIRE(cudaMemcpy(const_cast<void *>(host_A[s]), h_inputs[s].data(), m * ctx.vision_hidden_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaMemcpy(const_cast<void *>(host_B[s]), h_weights[s].data(), ctx.local_output_dim * ctx.vision_hidden_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice) == cudaSuccess);
    }

    launch_fused_multimodal_projection(
        host_A.data(), host_B.data(), host_D.data(), ctx.d_bias, nullptr, ctx.h_shapes.data(),
        ctx.num_segments, ctx.vision_hidden_size, ctx.text_hidden_size, ctx.tp_rank, ctx.tp_size,
        static_cast<int32_t>(DataType::BF16), nullptr
    );
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    for (int32_t s = 0; s < ctx.num_segments; ++s) {
        const int32_t m = ctx.h_shapes[s * 3 + 0];
        REQUIRE(cudaMemcpy(h_outputs[s].data(), host_D[s], m * ctx.local_output_dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost) == cudaSuccess);

        for (int32_t row = 0; row < m; ++row) {
            for (int32_t col = 0; col < ctx.local_output_dim; ++col) {
                float acc = 0.0f;
                for (int32_t k = 0; k < ctx.vision_hidden_size; ++k) {
                    acc += __bfloat162float(h_inputs[s][row * ctx.vision_hidden_size + k]) * __bfloat162float(h_weights[s][col * ctx.vision_hidden_size + k]);
                }
                acc += ctx.h_bias[ctx.tp_rank * ctx.local_output_dim + col];
                const float expected = acc / (1.0f + std::exp(-acc));
                const float actual = __bfloat162float(h_outputs[s][row * ctx.local_output_dim + col]);
                CHECK(actual == doctest::Approx(expected).epsilon(0.03));
            }
        }
        cudaFree(const_cast<void *>(host_A[s]));
        cudaFree(const_cast<void *>(host_B[s]));
        cudaFree(host_D[s]);
    }
}

TEST_CASE("MultimodalProjectionTest - TestFP8") {
    MultimodalProjectionContext ctx;

    std::vector<const void *> host_A(ctx.num_segments);
    std::vector<const void *> host_B(ctx.num_segments);
    std::vector<void *> host_D(ctx.num_segments);

    std::vector<std::vector<__nv_bfloat16> > h_inputs(ctx.num_segments);
    std::vector<std::vector<__nv_fp8_e4m3> > h_weights(ctx.num_segments);
    std::vector<std::vector<__nv_bfloat16> > h_outputs(ctx.num_segments);

    for (int32_t s = 0; s < ctx.num_segments; ++s) {
        const int32_t m = ctx.h_shapes[s * 3 + 0];
        h_inputs[s].assign(m * ctx.vision_hidden_size, __float2bfloat16(0.15f));
        h_weights[s].assign(ctx.local_output_dim * ctx.vision_hidden_size, static_cast<__nv_fp8_e4m3>(0.5f));
        h_outputs[s].resize(m * ctx.local_output_dim);

        REQUIRE(cudaMalloc(const_cast<void **>(&host_A[s]), m * ctx.vision_hidden_size * sizeof(__nv_bfloat16)) == cudaSuccess);
        REQUIRE(cudaMalloc(const_cast<void **>(&host_B[s]), ctx.local_output_dim * ctx.vision_hidden_size * sizeof(__nv_fp8_e4m3)) == cudaSuccess);
        REQUIRE(cudaMalloc(&host_D[s], m * ctx.local_output_dim * sizeof(__nv_bfloat16)) == cudaSuccess);

        REQUIRE(cudaMemcpy(const_cast<void *>(host_A[s]), h_inputs[s].data(), m * ctx.vision_hidden_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaMemcpy(const_cast<void *>(host_B[s]), h_weights[s].data(), ctx.local_output_dim * ctx.vision_hidden_size * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice) == cudaSuccess);
    }

    launch_fused_multimodal_projection(
        host_A.data(), host_B.data(), host_D.data(), ctx.d_bias, ctx.d_scales, ctx.h_shapes.data(),
        ctx.num_segments, ctx.vision_hidden_size, ctx.text_hidden_size, ctx.tp_rank, ctx.tp_size,
        static_cast<int32_t>(DataType::FP8), nullptr
    );
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    for (int32_t s = 0; s < ctx.num_segments; ++s) {
        const int32_t m = ctx.h_shapes[s * 3 + 0];
        REQUIRE(cudaMemcpy(h_outputs[s].data(), host_D[s], m * ctx.local_output_dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost) == cudaSuccess);

        const float current_scale = ctx.h_scales[s];
        for (int32_t row = 0; row < m; ++row) {
            for (int32_t col = 0; col < ctx.local_output_dim; ++col) {
                float acc = 0.0f;
                for (int32_t k = 0; k < ctx.vision_hidden_size; ++k) {
                    const float w_unpacked = static_cast<float>(h_weights[s][col * ctx.vision_hidden_size + k]) * current_scale;
                    acc += __bfloat162float(h_inputs[s][row * ctx.vision_hidden_size + k]) * w_unpacked;
                }
                acc += ctx.h_bias[ctx.tp_rank * ctx.local_output_dim + col];
                const float expected = acc / (1.0f + std::exp(-acc));
                const float actual = __bfloat162float(h_outputs[s][row * ctx.local_output_dim + col]);

                CHECK(actual == doctest::Approx(expected).epsilon(0.03));
            }
        }
        cudaFree(const_cast<void *>(host_A[s]));
        cudaFree(const_cast<void *>(host_B[s]));
        cudaFree(host_D[s]);
    }
}

TEST_CASE("MultimodalProjectionTest - TestFP4") {
    MultimodalProjectionContext ctx;

    std::vector<const void *> host_A(ctx.num_segments);
    std::vector<const void *> host_B(ctx.num_segments);
    std::vector<void *> host_D(ctx.num_segments);

    std::vector<std::vector<__nv_bfloat16> > h_inputs(ctx.num_segments);
    std::vector<std::vector<uint32_t> > h_weights_packed(ctx.num_segments);
    std::vector<std::vector<__nv_bfloat16> > h_outputs(ctx.num_segments);

    for (int32_t s = 0; s < ctx.num_segments; ++s) {
        const int32_t m = ctx.h_shapes[s * 3 + 0];
        h_inputs[s].assign(m * ctx.vision_hidden_size, __float2bfloat16(0.2f));

        const int32_t total_elements = ctx.local_output_dim * ctx.vision_hidden_size;
        h_weights_packed[s].assign(total_elements / 8, 0x11111111);
        h_outputs[s].resize(m * ctx.local_output_dim);

        REQUIRE(cudaMalloc(const_cast<void **>(&host_A[s]), m * ctx.vision_hidden_size * sizeof(__nv_bfloat16)) == cudaSuccess);
        REQUIRE(cudaMalloc(const_cast<void **>(&host_B[s]), (total_elements / 8) * sizeof(uint32_t)) == cudaSuccess);
        REQUIRE(cudaMalloc(&host_D[s], m * ctx.local_output_dim * sizeof(__nv_bfloat16)) == cudaSuccess);

        REQUIRE(cudaMemcpy(const_cast<void *>(host_A[s]), h_inputs[s].data(), m * ctx.vision_hidden_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaMemcpy(const_cast<void *>(host_B[s]), h_weights_packed[s].data(), (total_elements / 8) * sizeof(uint32_t), cudaMemcpyHostToDevice) == cudaSuccess);
    }

    launch_fused_multimodal_projection(
        host_A.data(), host_B.data(), host_D.data(), ctx.d_bias, ctx.d_scales, ctx.h_shapes.data(),
        ctx.num_segments, ctx.vision_hidden_size, ctx.text_hidden_size, ctx.tp_rank, ctx.tp_size,
        static_cast<int32_t>(DataType::FP4), nullptr
    );
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    for (int32_t s = 0; s < ctx.num_segments; ++s) {
        const int32_t m = ctx.h_shapes[s * 3 + 0];
        REQUIRE(cudaMemcpy(h_outputs[s].data(), host_D[s], m * ctx.local_output_dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost) == cudaSuccess);

        const float current_scale = ctx.h_scales[s];
        __half2_raw raw_h2 = __nv_cvt_fp4x2_to_halfraw2(0x11, __NV_E2M1);
        const __half2 h2 = *reinterpret_cast<__half2 *>(&raw_h2);
        const float fp4_value = __half2float(h2.x) * current_scale;

        for (int32_t row = 0; row < m; ++row) {
            for (int32_t col = 0; col < ctx.local_output_dim; ++col) {
                float acc = 0.0f;
                for (int32_t k = 0; k < ctx.vision_hidden_size; ++k) {
                    acc += __bfloat162float(h_inputs[s][row * ctx.vision_hidden_size + k]) * fp4_value;
                }
                acc += ctx.h_bias[ctx.tp_rank * ctx.local_output_dim + col];
                const float expected = acc / (1.0f + std::exp(-acc));
                const float actual = __bfloat162float(h_outputs[s][row * ctx.local_output_dim + col]);

                CHECK(actual == doctest::Approx(expected).epsilon(0.05));
            }
        }
        cudaFree(const_cast<void *>(host_A[s]));
        cudaFree(const_cast<void *>(host_B[s]));
        cudaFree(host_D[s]);
    }
}
