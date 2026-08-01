#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cuda_bf16.h>
#include <stdio.h>
#include <vector>

using namespace cute;

__global__ void cute_blackwell_bf16_kernel(
    float *c_matrix,
    const __nv_bfloat16 *a_matrix,
    const __nv_bfloat16 *b_matrix
) {
    if (threadIdx.x == 0) {
        printf("[DEVICE KERNEL] >>> SUCCESS! Control reached inside GPU kernel. <<<\n");
        printf("[DEVICE KERNEL] STEP 1: GPU Pointers received -> A: %p, B: %p, C: %p\n", a_matrix, b_matrix, c_matrix);
    }

    using mma_op = SM80_16x8x16_F32BF16BF16F32_TN;
    constexpr MMA_Atom<mma_op> mma_atom;
    constexpr auto tiled_mma = make_tiled_mma(mma_atom);

    const auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

    constexpr auto layout_A = make_layout(make_shape(Int<16>{}, Int<16>{}), LayoutRight{});
    constexpr auto layout_B = make_layout(make_shape(Int<16>{}, Int<16>{}), LayoutLeft{});
    constexpr auto layout_C = make_layout(make_shape(Int<16>{}, Int<16>{}), LayoutRight{});

    auto g_A = make_tensor(a_matrix, layout_A);
    auto g_B = make_tensor(b_matrix, layout_B);
    auto g_C = make_tensor(c_matrix, layout_C);

    auto tAgA = thr_mma.partition_A(g_A);
    auto tBgB = thr_mma.partition_B(g_B);
    auto tCgC = thr_mma.partition_C(g_C);

    auto tArA = thr_mma.make_fragment_A(tAgA);
    auto tBrB = thr_mma.make_fragment_B(tBgB);
    auto tCrC = thr_mma.make_fragment_C(tCgC);

    clear(tCrC);

    copy(tAgA, tArA);
    copy(tBgB, tBrB);

    __syncthreads();

    gemm(tiled_mma, tArA, tBrB, tCrC);

    __syncthreads();

    copy(tCrC, tCgC);
}

int main() {
    int device_id = 0;
    if (cudaSetDevice(device_id) != cudaSuccess) {
        printf("[HOST ERROR] Failed to set device 0\n");
        return -1;
    }

    cudaFree(0);

    constexpr int M = 16;
    constexpr int N = 16;
    constexpr int K = 16;

    std::vector<__nv_bfloat16> h_A(M * K);
    std::vector<__nv_bfloat16> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);

    for (int i = 0; i < M * K; ++i) h_A[i] = __float2bfloat16(1.0f);
    for (int i = 0; i < K * N; ++i) h_B[i] = __float2bfloat16(2.0f);

    float *d_C = nullptr;
    void *d_A = nullptr;
    void *d_B = nullptr;

    cudaMalloc(&d_A, h_A.size() * sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, h_B.size() * sizeof(__nv_bfloat16));
    cudaMalloc(&d_C, h_C.size() * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, h_C.size() * sizeof(float));

    printf("[HOST] Launching C++ CuTe GEMM kernel on Tensor Cores...\n");

    cute_blackwell_bf16_kernel<<<1, 32>>>(
        d_C,
        static_cast<const __nv_bfloat16 *>(d_A),
        static_cast<const __nv_bfloat16 *>(d_B)
    );

    cudaError_t launch_err = cudaGetLastError();
    printf("[HOST] Launch status: %d (%s)\n", launch_err, cudaGetErrorString(launch_err));

    cudaError_t sync_err = cudaDeviceSynchronize();
    printf("[HOST] Sync status: %d (%s)\n", sync_err, cudaGetErrorString(sync_err));

    cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\n=== [VERIFICATION] ===\n");
    printf("Expected cell value: 32.000000\n");
    printf("Actual cell value:   %f\n", h_C[0]);
    printf("=========================================\n\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

void deviceInfo() {
    printf("\n=========================================\n");
    printf("=== [HARDWARE CAPABILITIES REPORT] ===\n");
    printf("=========================================\n");

    int device_id = 0;
    if (cudaSetDevice(device_id) != cudaSuccess) {
        printf("[HOST ERROR] Failed to set CUDA device 0\n");
        return;
    }

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess) {
        printf("[HOST ERROR] Failed to get device properties\n");
        return;
    }

    int clock_rate = 0;
    int mem_clock_rate = 0;
    int max_threads_per_sm = 0;
    int max_blocks_per_sm = 0;

    cudaDeviceGetAttribute(&clock_rate, cudaDevAttrClockRate, device_id);
    cudaDeviceGetAttribute(&mem_clock_rate, cudaDevAttrMemoryClockRate, device_id);
    cudaDeviceGetAttribute(&max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, device_id);
    cudaDeviceGetAttribute(&max_blocks_per_sm, cudaDevAttrMaxBlocksPerMultiprocessor, device_id);

    printf("Device Name:                              %s\n", prop.name);
    printf("Compute Capability:                       %d.%d\n", prop.major, prop.minor);
    printf("Total Global Memory:                      %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
    printf("Multiprocessor (SM) Count:                %d\n", prop.multiProcessorCount);
    printf("Core Clock Rate:                          %.2f GHz\n", clock_rate * 1e-6);
    printf("L2 Cache Size:                            %d MB\n", prop.l2CacheSize / (1024 * 1024));

    printf("\n--- Memory & Warp Execution Limits ---\n");
    printf("Total Constant Memory:                    %zu KB\n", prop.totalConstMem / 1024);
    printf("Shared Memory per Block (Default):        %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Shared Memory per Block (Opt-in Max):     %zu KB\n", prop.sharedMemPerBlockOptin / 1024);
    printf("Shared Memory per SM:                     %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("Registers per Block:                      %d\n", prop.regsPerBlock);
    printf("Registers per SM:                         %d\n", prop.regsPerMultiprocessor);
    printf("Warp Size:                                %d\n", prop.warpSize);
    printf("Max Threads per Block:                    %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads per SM:                       %d\n", max_threads_per_sm);
    printf("Max Blocks per SM:                        %d\n", max_blocks_per_sm);

    printf("\n--- Grid & Block Geometry Limits ---\n");
    printf("Max Block Dimensions:                     [%d, %d, %d]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max Grid Dimensions:                      [%d, %d, %d]\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    printf("\n--- Hardware Engine Features ---\n");
    printf("Concurrent Kernels Support:               %s\n", prop.concurrentKernels ? "Yes" : "No");
    printf("Async Engine Count:                       %d\n", prop.asyncEngineCount);
    printf("Unified Addressing (UVA) Support:         %s\n", prop.unifiedAddressing ? "Yes" : "No");
    printf("Memory Clock Rate:                        %.2f GHz\n", mem_clock_rate * 1e-6);
    printf("Memory Bus Width:                         %d bits\n", prop.memoryBusWidth);
    printf("Peak Memory Bandwidth:                    %.2f GB/s\n", 2.0 * mem_clock_rate * (prop.memoryBusWidth / 8.0) * 1e-6);
    printf("Cooperative Launch Support:               %s\n", prop.cooperativeLaunch ? "Yes" : "No");
    printf("Multi-GPU Board:                          %s\n", prop.isMultiGpuBoard ? "Yes" : "No");
}
