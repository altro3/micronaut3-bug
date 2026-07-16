#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BLOCK_M 64
#define BLOCK_N 64

using namespace nvcuda::wmma;

template<int HEAD_DIM>
__global__ void flash_attention_tensor_core_prefill_kernel(
    half * __restrict__ output,
    const half * __restrict__ query,
    const half * __restrict__ key,
    const half * __restrict__ value,
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

    half *s_q = reinterpret_cast<half *>(s_dynamic_buffer);
    half *s_k = s_q + BLOCK_M * HEAD_DIM;
    half *s_v = s_k + BLOCK_N * HEAD_DIM;

    float *s_row_m = reinterpret_cast<float *>(s_v + BLOCK_N * HEAD_DIM);
    float *s_row_l = s_row_m + BLOCK_M;
    float *s_acc_o = s_row_l + BLOCK_M;

    float *s_tile_scores = s_acc_o + BLOCK_M * HEAD_DIM;
    half *s_tile_scores_half = reinterpret_cast<half *>(s_tile_scores + BLOCK_M * BLOCK_N);

    for (int m = tid; m < BLOCK_M; ++m) {
        s_row_m[m] = -1e20f;
        s_row_l[m] = 0.0f;
        for (int d = 0; d < HEAD_DIM; ++d) {
            s_acc_o[m * HEAD_DIM + d] = 0.0f;
        }
    }
    __syncthreads();

    const int start_m = tile_m_idx * BLOCK_M;
    const int q_stride = num_heads * HEAD_DIM;
    const int kv_stride = num_kv_heads * HEAD_DIM;

    for (int i = tid; i < BLOCK_M * HEAD_DIM; i += blockDim.x) {
        const int m = i / HEAD_DIM;
        const int d = i % HEAD_DIM;
        if (start_m + m < q_seq_len) {
            s_q[m * HEAD_DIM + d] = query[(start_m + m) * q_stride + q_head_idx * HEAD_DIM + d];
        } else {
            s_q[m * HEAD_DIM + d] = __float2half(0.0f);
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
                s_v[n * HEAD_DIM + d] = value[(start_n + n) * kv_stride + kv_head_idx * HEAD_DIM + d];
            } else {
                s_k[d * BLOCK_N + n] = __float2half(0.0f);
                s_v[n * HEAD_DIM + d] = __float2half(0.0f);
            }
        }
        __syncthreads();

        for (int wm = warp_id * WMMA_M; wm < BLOCK_M; wm += num_warps * WMMA_M) {
            for (int wn = 0; wn < BLOCK_N; wn += WMMA_N) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> q_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> k_frag;
                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

                fill_fragment(c_frag, 0.0f);

                for (int dk = 0; dk < HEAD_DIM; dk += WMMA_K) {
                    load_matrix_sync(q_frag, s_q + wm * HEAD_DIM + dk, HEAD_DIM);
                    load_matrix_sync(k_frag, s_k + dk * BLOCK_N + wn, BLOCK_N);
                    mma_sync(c_frag, q_frag, k_frag, c_frag);
                }

                store_matrix_sync(s_tile_scores + wm * BLOCK_N + wn, c_frag, BLOCK_N, mem_row_major);
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
                s_tile_scores_half[m * BLOCK_N + n] = __float2half(scores[n]);
            }

            const float scale_old = expf(old_m - new_m);
            s_row_l[m] = s_row_l[m] * scale_old + p_sum;
            s_row_m[m] = new_m;

            for (int d = lane_id; d < HEAD_DIM; d += 32) {
                s_acc_o[m * HEAD_DIM + d] *= scale_old;
            }
        }
        __syncthreads();

        for (int wm = warp_id * WMMA_M; wm < BLOCK_M; wm += num_warps * WMMA_M) {
            for (int wd = 0; wd < HEAD_DIM; wd += WMMA_N) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> s_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> v_frag;
                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> out_frag;

                load_matrix_sync(out_frag, s_acc_o + wm * HEAD_DIM + wd, HEAD_DIM, mem_row_major);

                for (int dk = 0; dk < BLOCK_N; dk += WMMA_K) {
                    load_matrix_sync(s_frag, s_tile_scores_half + wm * BLOCK_N + dk, BLOCK_N);
                    load_matrix_sync(v_frag, s_v + dk * HEAD_DIM + wd, HEAD_DIM);
                    mma_sync(out_frag, s_frag, v_frag, out_frag);
                }

                store_matrix_sync(s_acc_o + wm * HEAD_DIM + wd, out_frag, HEAD_DIM, mem_row_major);
            }
        }
        __syncthreads();
    }

    for (int m = warp_id; m < BLOCK_M; m += num_warps) {
        if (start_m + m < q_seq_len) {
            const float inv_l = 1.0f / (s_row_l[m] + 1e-9f);
            for (int d = lane_id; d < HEAD_DIM; d += 32) {
                output[(start_m + m) * q_stride + q_head_idx * HEAD_DIM + d] = __float2half(s_acc_o[m * HEAD_DIM + d] * inv_l);
            }
        }
    }
}

extern "C" {
void launch_flash_attention_prefill_fp16(
    half *output,
    const half *query,
    const half *key,
    const half *value,
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
        const auto k_kernel = flash_attention_tensor_core_prefill_kernel<256>;
        cudaFuncSetAttribute(k_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
        k_kernel<<<num_heads * batch_size, threads_per_block, shared_mem_size, stream>>>(output, query, key, value, q_seq_len, kv_seq_len, num_heads, num_kv_heads, scale);
    } else if (head_dim == 128) {
        const auto k_kernel = flash_attention_tensor_core_prefill_kernel<128>;
        cudaFuncSetAttribute(k_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
        k_kernel<<<num_heads * batch_size, threads_per_block, shared_mem_size, stream>>>(out_ptr, q_ptr, k_ptr, v_ptr, q_seq_len, kv_seq_len, num_heads, num_kv_heads, scale);
    } else if (head_dim == 64) {
        const auto k_kernel = flash_attention_tensor_core_prefill_kernel<64>;
        cudaFuncSetAttribute(k_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
        k_kernel<<<num_heads * batch_size, threads_per_block, shared_mem_size, stream>>>(out_ptr, q_ptr, k_ptr, v_ptr, q_seq_len, kv_seq_len, num_heads, num_kv_heads, scale);
    }
}
}
