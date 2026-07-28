#include <doctest/doctest.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include "cutlass/bfloat16.h"
#include "cutlass/float8.h"
#include "cutlass/float_subbyte.h"
#include "prefill_stage/dequantize_fusion_mma_gguf_q4_k_blackwell.cuh"

using ElementSFA = cutlass::float_ue4m3_t;
using ElementSFB = cutlass::float_ue4m3_t;
using ElementD = cutlass::bfloat16_t;

TEST_CASE("BlackwellNativeFp4GemmTest - Verification") {
    // Конфигурация под строго один аппаратный тайл Blackwell K=768
    const int32_t M = 128;
    const int32_t N = 256;
    const int32_t K = 768;

    // 1. Выделяем плоские векторы на хосте (1 байт = 2 элемента FP4)
    std::vector<uint8_t> h_A((M * K) / 2, 0);
    std::vector<uint8_t> h_B((N * K) / 2, 0);

    // Шкалы и выходная матрица в нативных типах CUTLASS
    std::vector<ElementSFA> h_SFA((M * K) / 16, ElementSFA(1.0f));
    std::vector<ElementSFB> h_SFB((N * K) / 16, ElementSFB(1.0f));
    std::vector<ElementD> h_D(M * N, ElementD(0.0f));

    // Заполняем матрицы А и B маской 0x77 (два элемента FP4 со взведенными битами в каждом байте)
    // Это исключает аппаратную интерпретацию маски как NaN/Zero на тензорных ядрах
    std::fill(h_A.begin(), h_A.end(), 0x77);
    std::fill(h_B.begin(), h_B.end(), 0x77);

    // Создаем выделенный поток исполнения (Stream)
    cudaStream_t test_stream;
    REQUIRE(cudaStreamCreate(&test_stream) == cudaSuccess);

    // Функция для округления размера вверх до выравнивания по кэш-линии Blackwell (128 байт)
    auto align_to_cache_line = [](size_t size) {
        return ((size + 127) / 128) * 128;
    };

    size_t size_A = align_to_cache_line(h_A.size());
    size_t size_B = align_to_cache_line(h_B.size());
    size_t size_SFA = align_to_cache_line(h_SFA.size() * sizeof(ElementSFA));
    size_t size_SFB = align_to_cache_line(h_SFB.size() * sizeof(ElementSFB));
    size_t size_D = align_to_cache_line(h_D.size() * sizeof(ElementD));

    void *d_A = nullptr;
    void *d_B = nullptr;
    void *d_SFA = nullptr;
    void *d_SFB = nullptr;
    void *d_D = nullptr;

    // Выделяем выровненную виртуальную память через асинхронный аллокатор CUDA
    // Это критично для корректного кодирования cuTensorMapEncode внутри CUTLASS initialize()
    REQUIRE(cudaMallocAsync(&d_A, size_A, test_stream) == cudaSuccess);
    REQUIRE(cudaMallocAsync(&d_B, size_B, test_stream) == cudaSuccess);
    REQUIRE(cudaMallocAsync(&d_SFA, size_SFA, test_stream) == cudaSuccess);
    REQUIRE(cudaMallocAsync(&d_SFB, size_SFB, test_stream) == cudaSuccess);
    REQUIRE(cudaMallocAsync(&d_D, size_D, test_stream) == cudaSuccess);

    // Асинхронное копирование данных на GPU
    REQUIRE(cudaMemcpyAsync(d_A, h_A.data(), h_A.size(), cudaMemcpyHostToDevice, test_stream) == cudaSuccess);
    REQUIRE(cudaMemcpyAsync(d_B, h_B.data(), h_B.size(), cudaMemcpyHostToDevice, test_stream) == cudaSuccess);
    REQUIRE(cudaMemcpyAsync(d_SFA, h_SFA.data(), h_SFA.size() * sizeof(ElementSFA), cudaMemcpyHostToDevice, test_stream) == cudaSuccess);
    REQUIRE(cudaMemcpyAsync(d_SFB, h_SFB.data(), h_SFB.size() * sizeof(ElementSFB), cudaMemcpyHostToDevice, test_stream) == cudaSuccess);
    REQUIRE(cudaMemsetAsync(d_D, 0, size_D, test_stream) == cudaSuccess);

    // Запускаем наше Blackwell TMA ядро на выделенном стриме
    launch_blackwell_fp4_native_gemm(
        d_D,
        d_A,
        d_B,
        d_SFA,
        d_SFB,
        M, N, K,
        static_cast<void *>(test_stream)
    );

    // Синхронизируем стрим и выкачиваем результаты
    REQUIRE(cudaStreamSynchronize(test_stream) == cudaSuccess);
    REQUIRE(cudaMemcpyAsync(h_D.data(), d_D, h_D.size() * sizeof(ElementD), cudaMemcpyDeviceToHost, test_stream) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(test_stream) == cudaSuccess);

    // Вытаскиваем центральный элемент для проверки факта вычислений тензорных ядер
    float sample_actual = static_cast<float>(h_D[0]);
    std::cout << "[ENGINE INFO] Blackwell Hardware MMA Output: " << sample_actual << std::endl;

    bool has_error = false;
    if (sample_actual == 0.0f) {
        std::cerr << "[ERROR] Kernel executed but returned absolute zeros. Tensor Cores are bypassed!" << std::endl;
        has_error = true;
    }

    CHECK_FALSE(has_error);

    // Асинхронная очистка ресурсов
    cudaFreeAsync(d_A, test_stream);
    cudaFreeAsync(d_B, test_stream);
    cudaFreeAsync(d_SFA, test_stream);
    cudaFreeAsync(d_SFB, test_stream);
    cudaFreeAsync(d_D, test_stream);
    cudaStreamDestroy(test_stream);
}
