#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <string_view>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

class L2CacheFlusher {
    void *d_flush_buffer = nullptr;
    size_t buffer_size = 0;

public:
    explicit L2CacheFlusher(const size_t size_bytes = 128 * 1024 * 1024) : buffer_size(size_bytes) {
        CUDA_CHECK(cudaMalloc(&d_flush_buffer, buffer_size));
    }

    ~L2CacheFlusher() {
        if (d_flush_buffer) {
            cudaFree(d_flush_buffer);
        }
    }

    void flush(const cudaStream_t stream = nullptr) const {
        if (d_flush_buffer && buffer_size > 0) {
            CUDA_CHECK(cudaMemsetAsync(d_flush_buffer, 0xAB, buffer_size, stream));
        }
    }

    L2CacheFlusher(const L2CacheFlusher &) = delete;

    L2CacheFlusher &operator=(const L2CacheFlusher &) = delete;
};

class GPUTimer {
    cudaEvent_t start_event;
    cudaEvent_t stop_event;

public:
    GPUTimer() {
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
    }

    ~GPUTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start(const cudaStream_t stream = nullptr) const {
        CUDA_CHECK(cudaEventRecord(start_event, stream));
    }

    void stop(const cudaStream_t stream = nullptr) const {
        CUDA_CHECK(cudaEventRecord(stop_event, stream));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    float elapsed_ms() const {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));
        return ms;
    }
};

struct BenchmarkCase {
    std::string_view name;

    void (*func)();
};

extern std::vector<BenchmarkCase> &get_benchmark_registry();

class BenchmarkRegistrar {
public:
    BenchmarkRegistrar(const std::string_view name, void (*func)()) {
        get_benchmark_registry().push_back({name, func});
    }
};

#define REGISTER_BENCHMARK(name, func) \
    static BenchmarkRegistrar const unique_registrar_##func(name, func)
