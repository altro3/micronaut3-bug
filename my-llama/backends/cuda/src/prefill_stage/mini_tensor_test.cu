#include "mini_tensor_test.cuh"
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cuda_bf16.h>

using namespace cute;

__global__ void cute_blackwell_bf16_kernel(
float *c_matrix,
const __nv_bfloat16 *a_matrix,
const __nv_bfloat16 *b_matrix
) {
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

extern "C" KERNEL_API void launch_cute_blackwell_gemm(float *d_C, const void *d_A, const void *d_B) {
    cute_blackwell_bf16_kernel<<<1, 32>>>(
    d_C,
    static_cast<const __nv_bfloat16 *>(d_A),
    static_cast<const __nv_bfloat16 *>(d_B)
    );
}
