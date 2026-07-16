#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math.h>
#include <stdint.h>

#define WMMA_M 32
#define WMMA_N 8
#define WMMA_K 16

#define BLOCK_M 64
#define BLOCK_N 64

using namespace nvcuda::wmma;

template<int HEAD_DIM>
__global__ void flash_attention_fp8_blackwell_prefill_kernel(
    __nv_fp8_e4m3 * __restrict__ output,
    const __nv_fp8_e4m3 * __restrict__ query,
    const __nv_fp8_e4m3 * __restrict__ key,
    const __nv_fp8_e4m3 * __restrict__ value,
    const int q_seq_len,
    const int kv_seq_len,
    const int num_heads,
    const int num_kv_heads,
    const float scale
) {
    const int q_head_idx = blockIdx.x;
    const int tile_m_idx = blockIdx.y;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockDim.x / 32;

    const int kv_head_ratio = num_heads / num_kv_heads;
    const int kv_head_idx = q_head_idx / kv_head_ratio;

    extern __shared__ uint8_t s_dynamic_buffer[];

    auto s_q = reinterpret_cast<__nv_fp8_e4m3 *>(s_dynamic_buffer);

    const int s_q_size_aligned = BLOCK_M * HEAD_DIM + 15 & ~15;
    __nv_fp8_e4m3 *s_k = s_q + s_q_size_aligned;

    const int s_k_size_aligned = BLOCK_N * HEAD_DIM + 15 & ~15;
    __nv_fp8_e4m3 *s_v = s_k + s_k_size_aligned;

    const int s_v_size_aligned = BLOCK_N * HEAD_DIM + 15 & ~15;
    auto s_row_m = reinterpret_cast<float *>(s_v + s_v_size_aligned);
    float *s_row_l = s_row_m + BLOCK_M;
    auto s_acc_o = reinterpret_cast<int *>(s_row_l + BLOCK_M);

    const int s_acc_o_size_aligned = BLOCK_M * HEAD_DIM + 3 & ~3;
    auto s_tile_scores = reinterpret_cast<float *>(s_acc_o + s_acc_o_size_aligned);

    constexpr int s_tile_scores_size_aligned = BLOCK_M * BLOCK_N + 3 & ~3;
    auto s_tile_scores_fp8 = reinterpret_cast<__nv_fp8_e4m3 *>(s_tile_scores + s_tile_scores_size_aligned);

    for (int m = tid; m < BLOCK_M; ++m) {
        s_row_m[m] = -1e20f;
        s_row_l[m] = 0.0f;
        for (int d = 0; d < HEAD_DIM; ++d) {
            s_acc_o[m * HEAD_DIM + d] = 0;
        }
    }
    __syncthreads();

    const int start_m = tile_m_idx * BLOCK_M;
    const int q_stride = num_heads * HEAD_DIM;
    const int kv_stride = num_kv_heads * HEAD_DIM;

    __nv_fp8_e4m3 zero_fp8;
    zero_fp8.__x = 0;

    for (int i = tid; i < BLOCK_M * HEAD_DIM; i += blockDim.x) {
        const int m = i / HEAD_DIM;
        const int d = i % HEAD_DIM;
        if (start_m + m < q_seq_len) {
            s_q[m * HEAD_DIM + d] = query[(start_m + m) * q_stride + q_head_idx * HEAD_DIM + d];
        } else {
            s_q[m * HEAD_DIM + d] = zero_fp8;
        }
    }
    __syncthreads();

    const int num_n_tiles = (kv_seq_len + BLOCK_N - 1) / BLOCK_N;

    for (int tile_n_idx = 0; tile_n_idx < num_n_tiles; ++tile_n_idx) {
        const int start_n = tile_n_idx * BLOCK_N;

        for (int i = tid; i < BLOCK_N * HEAD_DIM; i += blockDim.x) {
            const int n = i / HEAD_DIM;
            const int d = i % HEAD_DIM;
            if (start_n + n < kv_seq_len) {
                s_k[d * BLOCK_N + n] = key[(start_n + n) * kv_stride + kv_head_idx * HEAD_DIM + d];
                s_v[d * BLOCK_N + n] = value[(start_n + n) * kv_stride + kv_head_idx * HEAD_DIM + d];
            } else {
                s_k[d * BLOCK_N + n] = zero_fp8;
                s_v[d * BLOCK_N + n] = zero_fp8;
            }
        }
        __syncthreads();

        for (int wm = warp_id * WMMA_M; wm < BLOCK_M; wm += num_warps * WMMA_M) {
            for (int wn = 0; wn < BLOCK_N; wn += WMMA_N) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, unsigned char, row_major> q_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, unsigned char, col_major> k_frag;
                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, int> c_frag;

                fill_fragment(c_frag, 0);

                for (int dk = 0; dk < HEAD_DIM; dk += WMMA_K) {
                    const auto q_ptr = reinterpret_cast<const unsigned char *>(s_q + wm * HEAD_DIM + dk);
                    const auto k_ptr = reinterpret_cast<const unsigned char *>(s_k + dk * BLOCK_N + wn);

                    load_matrix_sync(q_frag, q_ptr, HEAD_DIM);
                    load_matrix_sync(k_frag, k_ptr, BLOCK_N);
                    mma_sync(c_frag, q_frag, k_frag, c_frag);
                }

                store_matrix_sync(reinterpret_cast<int *>(s_tile_scores) + wm * BLOCK_N + wn, c_frag, BLOCK_N, mem_row_major);
            }
        }
        __syncthreads();

        for (int m = warp_id; m < BLOCK_M; m += num_warps) {
            if (start_m + m >= q_seq_len) continue;

            float local_tile_max = -1e20f;
            float scores[BLOCK_N];

            for (int n = 0; n < BLOCK_N; ++n) {
                if (start_n + n >= kv_seq_len || start_m + m < start_n + n) {
                    scores[n] = -1e20f;
                } else {
                    scores[n] = s_tile_scores[m * BLOCK_N + n] * scale;
                    local_tile_max = fmaxf(local_tile_max, scores[n]);
                }
            }

            const float old_m = s_row_m[m];
            const float new_m = fmaxf(old_m, local_tile_max);

            float p_sum = 0.0f;
            for (int n = 0; n < BLOCK_N; ++n) {
                if (scores[n] > -1e19f) {
                    scores[n] = expf(scores[n] - new_m);
                    p_sum += scores[n];
                } else {
                    scores[n] = 0.0f;
                }
                s_tile_scores_fp8[m * BLOCK_N + n] = __nv_fp8_e4m3(__half(scores[n]));
            }

            const float scale_old = expf(old_m - new_m);
            s_row_l[m] = s_row_l[m] * scale_old + p_sum;
            s_row_m[m] = new_m;

            for (int d = lane_id; d < HEAD_DIM; d += 32) {
                s_acc_o[m * HEAD_DIM + d] = __float2int_rn(static_cast<float>(s_acc_o[m * HEAD_DIM + d]) * scale_old);
            }
        }
        __syncthreads();

        for (int wm = warp_id * WMMA_M; wm < BLOCK_M; wm += num_warps * WMMA_M) {
            for (int wd = 0; wd < HEAD_DIM; wd += WMMA_N) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, unsigned char, row_major> s_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, unsigned char, col_major> v_frag;
                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, int> out_frag;

                int *s_acc_o_ptr = s_acc_o + wm * HEAD_DIM + wd;
                load_matrix_sync(out_frag, s_acc_o_ptr, HEAD_DIM, mem_row_major);

                for (int dk = 0; dk < BLOCK_N; dk += WMMA_K) {
                    const auto s_tile_scores_ptr = reinterpret_cast<const unsigned char *>(s_tile_scores_fp8 + wm * BLOCK_N + dk);
                    const auto s_v_ptr = reinterpret_cast<const unsigned char *>(s_v + dk * BLOCK_N + wd);

                    load_matrix_sync(s_frag, s_tile_scores_ptr, BLOCK_N);
                    load_matrix_sync(v_frag, s_v_ptr, BLOCK_N);
                    mma_sync(out_frag, s_frag, v_frag, out_frag);
                }

                store_matrix_sync(s_acc_o_ptr, out_frag, HEAD_DIM, mem_row_major);
            }
        }
        __syncthreads();
    }

    for (int m = warp_id; m < BLOCK_M; m += num_warps) {
        if (start_m + m < q_seq_len) {
            const float inv_l = 1.0f / (s_row_l[m] + 1e-9f);
            for (int d = lane_id; d < HEAD_DIM; d += 32) {
                const float final_float_val = static_cast<float>(s_acc_o[m * HEAD_DIM + d]) * inv_l;
                output[(start_m + m) * q_stride + q_head_idx * HEAD_DIM + d] = __nv_fp8_e4m3(__half(final_float_val));
            }
        }
    }
}

extern "C" {
void launch_flash_attention_prefill_fp8(
    __nv_fp8_e4m3 *output,
    const __nv_fp8_e4m3 *query,
    const __nv_fp8_e4m3 *key,
    const __nv_fp8_e4m3 *value,
    const int batch_size,
    const int q_seq_len,
    const int kv_seq_len,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int threads_per_block,
    const float scale,
    const size_t shared_mem_size,
    cudaStream_t stream
) {
    if (head_dim == 256) {
        const auto k_kernel = flash_attention_fp8_blackwell_prefill_kernel<256>;
        cudaFuncSetAttribute(k_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
        k_kernel<<<num_heads * batch_size, threads_per_block, shared_mem_size, stream>>>(output, query, key, value, q_seq_len, kv_seq_len, num_heads, num_kv_heads, scale);
    } else if (head_dim == 128) {
        const auto k_kernel = flash_attention_fp8_blackwell_prefill_kernel<128>;
        cudaFuncSetAttribute(k_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
        k_kernel<<<num_heads * batch_size, threads_per_block, shared_mem_size, stream>>>(output, query, key, value, q_seq_len, kv_seq_len, num_heads, num_kv_heads, scale);
    } else if (head_dim == 64) {
        const auto k_kernel = flash_attention_fp8_blackwell_prefill_kernel<64>;
        cudaFuncSetAttribute(k_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
        k_kernel<<<num_heads * batch_size, threads_per_block, shared_mem_size, stream>>>(output, query, key, value, q_seq_len, kv_seq_len, num_heads, num_kv_heads, scale);
    }
}
}
