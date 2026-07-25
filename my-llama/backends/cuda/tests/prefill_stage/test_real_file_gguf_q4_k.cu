#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>
#include <fstream>
#include <string>
#include "data_types.h"
#include "prefill_stage/dequantize_fusion_mma_gguf_q4_k.cuh"

extern "C" void launch_fused_gemm_gguf_q4_k(
    void *output_activations, const void *input_activations, const void *quantized_weights,
    int32_t batch_size_or_tokens, int32_t hidden_units_out, int32_t hidden_units_in,
    int32_t data_type, void *stream_ptr
);

TEST(GgufRealFileTest, TestQwen05BQ4K) {
    const std::string file_path = "D:\\!models\\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf";
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "[SKIP Real File Test] File not found at " << file_path << std::endl;
        SUCCEED();
        return;
    }

    uint32_t magic = 0;
    file.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    if (magic != 0x46554747) {
        FAIL() << "Invalid GGUF magic in real model file!";
    }

    constexpr int32_t M = 1;
    constexpr int32_t N = 896;
    constexpr int32_t K = 896;

    constexpr size_t total_weight_elements = N * K;
    constexpr size_t total_blocks = total_weight_elements / 256;
    std::vector<BlockQ4K> host_real_weights(total_blocks);

    file.seekg(1024 * 1024 * 2);
    file.read(reinterpret_cast<char *>(host_real_weights.data()), total_blocks * sizeof(BlockQ4K));
    file.close();

    std::vector<__nv_bfloat16> h_input(M * K);
    for (int32_t i = 0; i < M * K; ++i) {
        h_input[i] = __float2bfloat16(0.01f);
    }
    std::vector<__nv_bfloat16> h_output(M * N);

    void *d_out, *d_in, *d_w;
    ASSERT_EQ(cudaMalloc(&d_out, M * N * sizeof(__nv_bfloat16)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_in, M * K * sizeof(__nv_bfloat16)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_w, host_real_weights.size() * sizeof(BlockQ4K)), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_in, h_input.data(), h_input.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_w, host_real_weights.data(), host_real_weights.size() * sizeof(BlockQ4K), cudaMemcpyHostToDevice), cudaSuccess);

    launch_fused_gemm_gguf_q4_k(d_out, d_in, d_w, M, N, K, static_cast<int32_t>(DataType::BF16), nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(h_output.data(), d_out, h_output.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), cudaSuccess);

    std::cout << "[REAL QWEN 0.5B TEST] Execution successful on sm_120!" << std::endl;
    std::cout << "[REAL QWEN 0.5B TEST] First 5 activation logits: " << std::endl;
    for (int32_t i = 0; i < 5; ++i) {
        std::cout << "  logit[" << i << "] = " << __bfloat162float(h_output[i]) << std::endl;
    }

    cudaFree(d_out);
    cudaFree(d_in);
    cudaFree(d_w);
}
