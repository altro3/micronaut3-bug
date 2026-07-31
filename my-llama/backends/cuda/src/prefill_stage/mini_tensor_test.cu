#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cuda_bf16.h>
#include <stdio.h>

using namespace cute;

__global__ void cute_blackwell_bf16_kernel(
    float *c_matrix,
    const __nv_bfloat16 *a_matrix,
    const __nv_bfloat16 *b_matrix
) {
    if (threadIdx.x == 0) {
        printf("[DEVICE KERNEL] STEP 1: Entered kernel. GPU Pointers -> A: %p, B: %p, C: %p\n", a_matrix, b_matrix, c_matrix);
    }

    using mma_op = SM80_16x8x16_F32BF16BF16F32_TN;

    if (threadIdx.x == 0) {
        printf("[DEVICE KERNEL] STEP 2: Instantiating MMA_Atom<mma_op>\n");
    }
    constexpr MMA_Atom<mma_op> mma_atom;

    if (threadIdx.x == 0) {
        printf("[DEVICE KERNEL] STEP 3: Executing make_tiled_mma\n");
    }
    constexpr auto tiled_mma = make_tiled_mma(mma_atom);

    if (threadIdx.x == 0) {
        printf("[DEVICE KERNEL] STEP 4: Fetching thread slice via get_thread_slice\n");
    }
    const auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

    if (threadIdx.x == 0) {
        printf("[DEVICE KERNEL] STEP 5: Declaring matrix shapes and layouts (16x16)\n");
    }
    constexpr auto layout_A = make_layout(make_shape(Int<16>{}, Int<16>{}), GenRowMajor{});
    constexpr auto layout_B = make_layout(make_shape(Int<16>{}, Int<16>{}), GenColMajor{});
    constexpr auto layout_C = make_layout(make_shape(Int<16>{}, Int<16>{}), GenRowMajor{});

    if (threadIdx.x == 0) {
        printf("[DEVICE KERNEL] STEP 6: Creating global memory tensors\n");
    }
    auto g_A = make_tensor(a_matrix, layout_A);
    auto g_B = make_tensor(b_matrix, layout_B);
    auto g_C = make_tensor(c_matrix, layout_C);

    if (threadIdx.x == 0) {
        printf("[DEVICE KERNEL] STEP 7: Partitioning global tensors per thread lane\n");
    }
    auto tAgA = thr_mma.partition_A(g_A);
    auto tBgB = thr_mma.partition_B(g_B);
    auto tCgC = thr_mma.partition_C(g_C);

    if (threadIdx.x == 0) {
        printf("[DEVICE KERNEL] STEP 8: Allocating hardware register fragments\n");
    }
    auto tArA = thr_mma.make_fragment_A(tAgA);
    auto tBrB = thr_mma.make_fragment_B(tBgB);
    auto tCrC = thr_mma.make_fragment_C(tCgC);

    if (threadIdx.x == 0) {
        printf("[DEVICE KERNEL] STEP 9: Clearing accumulator registers\n");
    }
    clear(tCrC);

    if (threadIdx.x == 0) {
        printf("[DEVICE KERNEL] STEP 10: Copying data from GMEM to registers (A and B)\n");
    }
    copy(tAgA, tArA);
    copy(tBgB, tBrB);

    printf("[DEVICE KERNEL] Thread %d reached pre-GEMM barrier\n", threadIdx.x);
    __syncthreads();

    if (threadIdx.x == 0) {
        printf("[DEVICE KERNEL] STEP 11: Invoking hardware cute::gemm with MMA_Atom signature\n");
    }
    cute::gemm(mma_atom, tArA, tBrB, tCrC);

    printf("[DEVICE KERNEL] Thread %d reached post-GEMM barrier\n", threadIdx.x);
    __syncthreads();

    if (threadIdx.x == 0) {
        printf("[DEVICE KERNEL] STEP 12: Copying results back from registers to global memory C\n");
    }
    copy(tCrC, tCgC);

    if (threadIdx.x == 0) {
        printf("[DEVICE KERNEL] STEP 13: Kernel execution pipeline fully completed.\n");
    }
}

extern "C" cudaError_t launch_mini_wmma_bf16(
    float *d_C,
    const void *d_A,
    const void *d_B
) {
    printf("[HOST LAUNCH] Inside launch_mini_wmma_bf16 function entry point\n");
    printf("[HOST LAUNCH] Matrix A: %p, Matrix B: %p, Matrix C: %p\n", d_A, d_B, d_C);

    const auto a_ptr = static_cast<const __nv_bfloat16 *>(d_A);
    const auto b_ptr = static_cast<const __nv_bfloat16 *>(d_B);

    printf("[HOST LAUNCH] Dispatching __global__ cute_blackwell_bf16_kernel<<<1, 32>>>\n");
    cute_blackwell_bf16_kernel<<<1, 32, 0, 0>>>(d_C, a_ptr, b_ptr);

    const cudaError_t err = cudaGetLastError();
    printf("[HOST LAUNCH] Driver evaluation status immediately after dispatch: %d\n", err);

    return err;
}

int main() {
    printf("\n=========================================\n");
    printf("=== [MONOLITHIC DIRECT KERNEL LAUNCH] ===\n");
    printf("=========================================\n");

    constexpr int M = 16;
    constexpr int N = 16;
    constexpr int K = 16;

    std::vector<__nv_bfloat16> h_A(M * K);
    std::vector<__nv_bfloat16> h_B(K * N);
    std::vector h_C(M * N, 0.0f);

    for (int i = 0; i < M * K; ++i) h_A[i] = __float2bfloat16(1.0f);
    for (int i = 0; i < K * N; ++i) h_B[i] = __float2bfloat16(2.0f);

    if (cudaSetDevice(0) != cudaSuccess) {
        printf("[HOST ERROR] Failed to set CUDA device 0\n");
        return -1;
    }

    float *d_C = nullptr;
    void *d_A = nullptr;
    void *d_B = nullptr;

    cudaMalloc(&d_A, h_A.size() * sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, h_B.size() * sizeof(__nv_bfloat16));
    cudaMalloc(&d_C, h_C.size() * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, h_C.size() * sizeof(float));

    printf("[HOST] Directly dispatching __global__ kernel bypassing any libraries...\n");

    cute_blackwell_bf16_kernel<<<1, 32, 0, 0>>>(d_C, static_cast<const __nv_bfloat16 *>(d_A), static_cast<const __nv_bfloat16 *>(d_B));

    const cudaError_t launch_err = cudaGetLastError();
    printf("[HOST] Launch error status code: %d (%s)\n", launch_err, cudaGetErrorString(launch_err));

    const cudaError_t sync_err = cudaDeviceSynchronize();
    printf("[HOST] Device sync status code: %d (%s)\n", sync_err, cudaGetErrorString(sync_err));

    cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\n=== [TENSOR CORE VALUE VERIFICATION] ===\n");
    printf("Expected cell value: 32\n");
    printf("Actual cell value:   %f\n", h_C[0]);
    printf("=========================================\n\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
