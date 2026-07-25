#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <iostream>

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    int deviceCount = 0;
    const cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "Error: No CUDA execution environment detected!" << std::endl;
        return -1;
    }
    cudaSetDevice(0);

    return RUN_ALL_TESTS();
}
