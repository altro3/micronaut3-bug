#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>
#include <cuda_runtime.h>
#include <iostream>

int main(const int argc, char **argv) {
    doctest::Context context;
    context.applyCommandLine(argc, argv);

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "Error: No CUDA execution environment detected!" << std::endl;
        return -1;
    }

    cudaSetDevice(0);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    std::cout << "[SYSTEM]: Testing on GPU: " << props.name << " (sm_" << props.major << props.minor << ")" << std::endl;

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "[MEMORY]: Free VRAM: " << free_mem / (1024 * 1024) << " MB / " << total_mem / (1024 * 1024) << " MB" << std::endl;

    cudaFree(0);

    const int res = context.run();

    cudaDeviceSynchronize();

    if (context.shouldExit()) {
        return res;
    }

    return res;
}
