#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string_view>
#include "bench_utils.cuh"

std::vector<BenchmarkCase> &get_benchmark_registry() {
    static std::vector<BenchmarkCase> registry;
    return registry;
}

int main(const int argc, char **argv) {
    int device_count = 0;
    const cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cerr << "Error: No CUDA execution environment detected!" << std::endl;
        return -1;
    }
    cudaSetDevice(0);

    bool run_all = true;
    std::string_view target_bench = "";

    if (argc > 1) {
        target_bench = argv[1];
        if (target_bench != "all") {
            run_all = false;
        }
    }

    const auto &benchmarks = get_benchmark_registry();
    std::cout << "[STARTING CUDA KERNEL BENCHMARKS] Registered count: " << benchmarks.size() << std::endl;

    size_t executed_count = 0;
    for (const auto &[name, func]: benchmarks) {
        if (run_all || name == target_bench) {
            std::cout << "\nExecuting benchmark: " << name << std::endl;
            func();
            executed_count++;
        }
    }

    if (executed_count == 0 && !run_all) {
        std::cerr << "Error: Benchmark '" << target_bench << "' not found in registry!" << std::endl;
        std::cerr << "Available benchmarks:" << std::endl;
        for (const auto &[name, func]: benchmarks) {
            std::cerr << "  - " << name << std::endl;
        }
        return -1;
    }

    std::cout << "\n[BENCHMARKS COMPLETED. Total executed: " << executed_count << "]" << std::endl;
    return 0;
}
