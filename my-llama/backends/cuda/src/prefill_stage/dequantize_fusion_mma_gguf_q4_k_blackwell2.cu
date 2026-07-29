#include <cuda_runtime.h>
#include <mma.h>
#include <iostream>

using namespace nvcuda;

__device__ __forceinline__ float unpack_fp4(uint32_t val) {
    uint32_t sign = (val & 0x8) << 28;
    uint32_t exp = (val & 0x6) >> 1;
    uint32_t mantissa = (val & 0x1);
    if (exp == 0 && mantissa == 0) return 0.0f;
    uint32_t bias_exp = (exp == 0) ? 0 : (exp + 126) << 23;
    return __int_as_float(sign | bias_exp | (mantissa << 22));
}

__global__ void __launch_bounds__(128) llama_style_fp4_gemm_kernel(
    half * __restrict__ output_d,
    const uint8_t * __restrict__ input_a,
    const uint8_t * __restrict__ weights_b,
    const float * __restrict__ scales_a,
    const float * __restrict__ scales_b,
    int M, int N, int K
) {
    __shared__ half smem_A[16][16];
    __shared__ half smem_B[16][16];

    int lnum = threadIdx.x;
    int warp_id = lnum / 32;

    int block_m = blockIdx.y * 16;
    int block_n = blockIdx.x * 16;

    if (block_m >= M || block_n >= N) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for (int k_block = 0; k_block < K; k_block += 16) {
        if (lnum < 16) {
            int m_idx = block_m + lnum;
#pragma unroll
            for (int k_offset = 0; k_offset < 16; ++k_offset) {
                int k_idx = k_block + k_offset;
                if (m_idx < M && k_idx < K) {
                    uint8_t pack = input_a[(m_idx * K + k_idx) / 2];
                    uint32_t val = (k_idx % 2 == 0) ? (pack >> 4) : (pack & 0x0F);
                    float scale = scales_a[m_idx * (K / 16) + (k_idx / 16)];
                    smem_A[lnum][k_offset] = __float2half(unpack_fp4(val) * scale);
                } else {
                    smem_A[lnum][k_offset] = __float2half(0.0f);
                }
            }
        }

        if (lnum >= 32 && lnum < 48) {
            int n_idx = block_n + (lnum - 32);
#pragma unroll
            for (int k_offset = 0; k_offset < 16; ++k_offset) {
                int k_idx = k_block + k_offset;
                if (n_idx < N && k_idx < K) {
                    uint8_t pack = weights_b[(n_idx * K + k_idx) / 2];
                    uint32_t val = (k_idx % 2 == 0) ? (pack >> 4) : (pack & 0x0F);
                    float scale = scales_b[n_idx * (K / 16) + (k_idx / 16)];
                    smem_B[lnum - 32][k_offset] = __float2half(unpack_fp4(val) * scale);
                } else {
                    smem_B[lnum - 32][k_offset] = __float2half(0.0f);
                }
            }
        }

        __syncthreads();

        wmma::load_matrix_sync(a_frag, (half *) smem_A, 16);
        wmma::load_matrix_sync(b_frag, (half *) smem_B, 16);

        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        __syncthreads();
    }

    if (warp_id == 0) {
        half *block_output = output_d + block_m * N + block_n;
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> out_frag;

#pragma unroll
        for (int i = 0; i < acc_frag.num_elements; ++i) {
            out_frag.x[i] = __float2half(acc_frag.x[i]);
        }
        wmma::store_matrix_sync(block_output, out_frag, N, wmma::mem_row_major);
    }
}

extern "C" void launch_blackwell_fp4_native_gemm(
    void *output_d,
    const void *input_a,
    const void *weights_b,
    const void *scales_a,
    const void *scales_b,
    int32_t m_extent,
    int32_t n_extent,
    int32_t k_extent,
    void *stream_ptr
) {
    int M = m_extent;
    int N = n_extent;
    int K = k_extent;
    auto stream = static_cast<cudaStream_t>(stream_ptr);

    if (M == 0 || N == 0 || K == 0) return;

    dim3 block(128, 1, 1);
    dim3 grid((N + 15) / 16, (M + 15) / 16, 1);

    llama_style_fp4_gemm_kernel<<<grid, block, 0, stream>>>(
        static_cast<half *>(output_d),
        static_cast<const uint8_t *>(input_a),
        static_cast<const uint8_t *>(weights_b),
        static_cast<const float *>(scales_a),
        static_cast<const float *>(scales_b),
        M, N, K
    );
}
