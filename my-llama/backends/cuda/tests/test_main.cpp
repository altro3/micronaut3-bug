#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>
#include <cuda_runtime.h>
#include <iostream>

int main(const int argc, char **argv) {
    doctest::Context context;
    context.applyCommandLine(argc, argv);

    int deviceCount = 0;
    const cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "Error: No CUDA execution environment detected!" << std::endl;
        return -1;
    }
    cudaSetDevice(0);

    const int res = context.run();
    if (context.shouldExit()) {
        return res;
    }

    return res;
}
