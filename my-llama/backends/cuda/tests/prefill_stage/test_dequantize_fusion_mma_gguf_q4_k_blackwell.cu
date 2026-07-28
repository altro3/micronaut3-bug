#include <doctest/doctest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include "cutlass/bfloat16.h"
#include "cutlass/float8.h"
#include "cutlass/float_subbyte.h"
#include "cutlass/layout/vector.h" // Тот самый пропущенный инклуд
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "prefill_stage/dequantize_fusion_mma_gguf_q4_k_blackwell.cuh"

using ElementSFA = cutlass::float_ue4m3_t;
using ElementSFB = cutlass::float_ue4m3_t;
using ElementD = cutlass::bfloat16_t;

TEST_CASE("BlackwellNativeFp4GemmTest - Verification") {
    const int32_t M = 256;
    const int32_t N = 1024;
    const int32_t K = 1024;

    cutlass::HostTensor<uint8_t, cutlass::layout::PackedVectorLayout> tensor_A_packed;
    cutlass::HostTensor<uint8_t, cutlass::layout::PackedVectorLayout> tensor_B_packed;

    cutlass::HostTensor<ElementSFA, cutlass::layout::PackedVectorLayout> tensor_SFA;
    cutlass::HostTensor<ElementSFB, cutlass::layout::PackedVectorLayout> tensor_SFB;
    cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> tensor_D;

    tensor_A_packed.reset(cutlass::make_Coord((M * K) / 2));
    tensor_B_packed.reset(cutlass::make_Coord((N * K) / 2));
    tensor_SFA.reset(cutlass::make_Coord((M * K) / 16));
    tensor_SFB.reset(cutlass::make_Coord((N * K) / 16));
    tensor_D.reset(cutlass::make_Coord(M * N));

    std::vector<float> h_raw_A(M * K, 0.5f);
    std::vector<float> h_raw_B(N * K, 0.25f);

    for (int32_t m = 0; m < M; ++m) {
        for (int32_t k = 0; k < K; k += 16) {
            float max_val = 0.0f;
            for (int32_t i = 0; i < 16; ++i) {
                max_val = std::max(max_val, std::abs(h_raw_A[m * K + k + i]));
            }
            float sf_a = max_val / 6.0f;
            if (sf_a == 0.0f) sf_a = 1.0f;

            tensor_SFA.host_data()[(m * K + k) / 16] = cutlass::float_ue4m3_t(sf_a);

            for (int32_t i = 0; i < 16; i += 2) {
                int32_t idx0 = m * K + k + i;
                int32_t idx1 = m * K + k + i + 1;

                float q_val0 = std::max(-6.0f, std::min(6.0f, std::round(h_raw_A[idx0] / sf_a)));
                float q_val1 = std::max(-6.0f, std::min(6.0f, std::round(h_raw_A[idx1] / sf_a)));

                uint8_t packed_byte = (static_cast<uint8_t>(q_val0) & 0x0F) |
                                      ((static_cast<uint8_t>(q_val1) & 0x0F) << 4);

                tensor_A_packed.host_data()[idx0 / 2] = packed_byte;
            }
        }
    }

    for (int32_t n = 0; n < N; ++n) {
        for (int32_t k = 0; k < K; k += 16) {
            float max_val = 0.0f;
            for (int32_t i = 0; i < 16; ++i) {
                max_val = std::max(max_val, std::abs(h_raw_B[n * K + k + i]));
            }
            float sf_b = max_val / 6.0f;
            if (sf_b == 0.0f) sf_b = 1.0f;

            tensor_SFB.host_data()[(n * K + k) / 16] = cutlass::float_ue4m3_t(sf_b);

            for (int32_t i = 0; i < 16; i += 2) {
                int32_t k_idx0 = k + i;
                int32_t k_idx1 = k + i + 1;

                float q_val0 = std::max(-6.0f, std::min(6.0f, std::round(h_raw_B[n * K + k_idx0] / sf_b)));
                float q_val1 = std::max(-6.0f, std::min(6.0f, std::round(h_raw_B[n * K + k_idx1] / sf_b)));

                uint8_t packed_byte = (static_cast<uint8_t>(q_val0) & 0x0F) |
                                      ((static_cast<uint8_t>(q_val1) & 0x0F) << 4);

                int32_t linear_packed_idx = (k_idx0 * N + n) / 2;
                tensor_B_packed.host_data()[linear_packed_idx] = packed_byte;
            }
        }
    }

    tensor_A_packed.sync_device();
    tensor_B_packed.sync_device();
    tensor_SFA.sync_device();
    tensor_SFB.sync_device();

    cudaMemset(tensor_D.device_data(), 0, M * N * sizeof(ElementD));

    launch_blackwell_fp4_native_gemm(
        tensor_D.device_data(),
        tensor_A_packed.device_data(),
        tensor_B_packed.device_data(),
        tensor_SFA.device_data(),
        tensor_SFB.device_data(),
        M, N, K,
        nullptr
    );

    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    tensor_D.sync_host();

    bool has_error = false;
    for (int32_t m = 0; m < M && !has_error; ++m) {
        for (int32_t n = 0; n < N && !has_error; ++n) {
            float expected = 0.0f;
            for (int32_t k = 0; k < K; ++k) {
                float sf_a = static_cast<float>(tensor_SFA.host_data()[(m * K + k) / 16]);
                float sf_b = static_cast<float>(tensor_SFB.host_data()[(n * K + k) / 16]);

                uint8_t byte_A = tensor_A_packed.host_data()[(m * K + k) / 2];
                int8_t raw_fp4_A = (k % 2 == 0) ? (byte_A & 0x0F) : (byte_A >> 4);
                if (raw_fp4_A > 7) raw_fp4_A -= 16;

                uint8_t byte_B = tensor_B_packed.host_data()[(k * N + n) / 2];
                int8_t raw_fp4_B = (k % 2 == 0) ? (byte_B & 0x0F) : (byte_B >> 4);
                if (raw_fp4_B > 7) raw_fp4_B -= 16;

                float val_a = static_cast<float>(raw_fp4_A) * sf_a;
                float val_b = static_cast<float>(raw_fp4_B) * sf_b;

                expected += val_a * val_b;
            }

            const float actual = static_cast<float>(tensor_D.host_data()[m * N + n]);

            if (std::abs(actual - expected) > 1.5f) {
                printf("[ERROR] Mismatch at M=%d, N=%d | Actual: %f, Expected: %f\n", m, n, actual, expected);
                has_error = true;
            }
        }
    }

    CHECK_FALSE(has_error);
}
