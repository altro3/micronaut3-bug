#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <string_view>
#include <algorithm>
#define CUDA_CHECK(call) \
do { \
cudaError_t err = call; \
if (err != cudaSuccess) { \
std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
<< " at " << __FILE__ << ":" << __LINE__ << std::endl; \
std::exit(EXIT_FAILURE); \
} \
} while (0)
template<typename T>
class DeviceBuffer {
T *d_ptr = nullptr;
size_t count = 0;

public:
explicit DeviceBuffer(const size_t element_count) : count(element_count) {
    if (count > 0) {
        CUDA_CHECK(cudaMalloc(&d_ptr, count * sizeof(T)));
    }
}

DeviceBuffer(const size_t element_count, const int memset_value) : count(element_count) {
    if (count > 0) {
        CUDA_CHECK(cudaMalloc(&d_ptr, count * sizeof(T)));
        CUDA_CHECK(cudaMemset(d_ptr, memset_value, count * sizeof(T)));
    }
}

explicit DeviceBuffer(const std::vector<T> &host_vec) : count(host_vec.size()) {
    if (count > 0) {
        CUDA_CHECK(cudaMalloc(&d_ptr, count * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(d_ptr, host_vec.data(), count * sizeof(T), cudaMemcpyHostToDevice));
    }
}

~DeviceBuffer() {
    if (d_ptr) {
        cudaFree(d_ptr);
    }
}

DeviceBuffer(DeviceBuffer &&other) noexcept : d_ptr(other.d_ptr), count(other.count) {
    other.d_ptr = nullptr;
    other.count = 0;
}

DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
    if (this != &other) {
        if (d_ptr) {
            cudaFree(d_ptr);
        }
        d_ptr = other.d_ptr;
        count = other.count;
        other.d_ptr = nullptr;
        other.count = 0;
    }
    return *this;
}

T *get() const { return d_ptr; }
void *get_void() const { return reinterpret_cast<void *>(d_ptr); }
const void *get_const_void() const { return reinterpret_cast<const void *>(d_ptr); }
size_t size_bytes() const { return count * sizeof(T); }

void copy_to_device(const std::vector<T> &host_vec) {
    const size_t copy_count = std::min(count, host_vec.size());
    if (copy_count > 0) {
        CUDA_CHECK(cudaMemcpy(d_ptr, host_vec.data(), copy_count * sizeof(T), cudaMemcpyHostToDevice));
    }
}

DeviceBuffer(const DeviceBuffer &) = delete;

DeviceBuffer &operator=(const DeviceBuffer &) = delete;
};

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

class BenchmarkReporter {
public:
    static void report_performance(
        const std::string_view target_name,
        const std::string_view type_name,
        std::vector<float> &iters_ms,
        const double total_fops = 0.0,
        const double total_bytes_moved = 0.0,
        const bool use_gflops_unit = false
    ) {
        if (iters_ms.empty()) return;
        std::ranges::sort(iters_ms);
        float total_time_ms = 0.0f;
        for (const float t: iters_ms) {
            total_time_ms += t;
        }
        const float avg_time_ms = total_time_ms / iters_ms.size();
        const float p50 = iters_ms[static_cast<size_t>(iters_ms.size() * 0.50)];
        const float p90 = iters_ms[static_cast<size_t>(iters_ms.size() * 0.90)];
        const float p95 = iters_ms[static_cast<size_t>(iters_ms.size() * 0.95)];
        std::cout << "==========================================================================" << std::endl;
        std::cout << "[" << target_name << " BENCHMARK RESULTS - " << type_name << "]" << std::endl;
        std::cout << "  Real Average Time:        " << avg_time_ms << " ms" << std::endl;
        std::cout << "  Percentile 50% (Median): " << p50 << " ms" << std::endl;
        std::cout << "  Percentile 90%:          " << p90 << " ms" << std::endl;
        std::cout << "  Percentile 95%:          " << p95 << " ms" << std::endl;
        if (total_fops > 0.0) {
            if (use_gflops_unit) {
                const double avg_gflops = total_fops * 1e-9 / (static_cast<double>(avg_time_ms) * 1e-3);
                std::cout << "  Average Compute Perf:    " << avg_gflops << " GFLOPs" << std::endl;
            } else {
                const double avg_tflops = total_fops * 1e-12 / (static_cast<double>(avg_time_ms) * 1e-3);
                if (avg_tflops >= 1000.0) {
                    std::cout << "  Real Compute Perf:       " << avg_tflops / 1000.0 << " PFLOPs/s" << std::endl;
                } else {
                    std::cout << "  Real Compute Perf:       " << avg_tflops << " TFLOPs/s" << std::endl;
}
}
}
if (total_bytes_moved > 0.0) {
const double avg_gb_s = total_bytes_moved * 1e-9 / (static_cast<double>(avg_time_ms) * 1e-3);
std::cout << "  Real Bandwidth:          " << avg_gb_s << " GB/s" << std::endl;
}
std::cout << "==========================================================================" << std::endl;
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
get_benchmark_registry().push_back({.name = name, .func = func});
}
};
#define REGISTER_BENCHMARK(name, func) \
static BenchmarkRegistrar const unique_registrar_##func(name, func)
