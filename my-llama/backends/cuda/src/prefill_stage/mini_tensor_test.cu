#include "mini_tensor_test.cuh"
#include <mma.h>
#include <cuda_bf16.h>
#include <iostream>

using namespace nvcuda::wmma;

__global__ void wmma_bf16_kernel(float *c_matrix, const __nv_bfloat16 *a_matrix, const __nv_bfloat16 *b_matrix) {
    int lda = 16;
    int ldb = 16;
    int ldc = 16;

    fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;

    fill_fragment(c_frag, 0.0f);

    load_matrix_sync(a_frag, a_matrix, lda);
    load_matrix_sync(b_frag, b_matrix, ldb);

    mma_sync(c_frag, a_frag, b_frag, c_frag);

    store_matrix_sync(c_matrix, c_frag, ldc, mem_row_major);
}

extern "C" cudaError_t launch_mini_wmma_bf16(
    float *d_C,
    const void *d_A,
    const void *d_B,
    cudaStream_t stream
) {
    const __nv_bfloat16 *a_ptr = static_cast<const __nv_bfloat16 *>(d_A);
    const __nv_bfloat16 *b_ptr = static_cast<const __nv_bfloat16 *>(d_B);

    wmma_bf16_kernel<<<1, 32, 0, stream>>>(d_C, a_ptr, b_ptr);

    return cudaGetLastError();
}
