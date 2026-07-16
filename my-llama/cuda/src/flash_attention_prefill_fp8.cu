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

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

template<int HEAD_DIM>
__global__ void flash_attention_fp8_blackwell_prefill_kernel(
    __nv_fp8_e4m3 * __restrict__ output,
    const __nv_fp8_e4m3 * __restrict__ query,
    const __nv_fp8_e4m3 * __restrict__ key,
    const __nv_fp8_e4m3 * __restrict__ value,
    int q_seq_len,
    int kv_seq_len,
    int num_heads,
    int num_kv_heads,
    float scale,
    int batch_size
) {
    const int batch_idx = blockIdx.x / num_heads;
    const int q_head_idx = blockIdx.x % num_heads;
    const int tile_m_idx = blockIdx.y;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockDim.x / 32;

    const int kv_head_ratio = num_heads / num_kv_heads;
    const int kv_head_idx = q_head_idx / kv_head_ratio;

    constexpr int PADDING = 16;
    const int stride_q = HEAD_DIM + PADDING;
    const int stride_k = HEAD_DIM + PADDING;
    const int stride_v = BLOCK_N + PADDING;

    extern __shared__ uint8_t s_dynamic_buffer[];

    auto s_q = reinterpret_cast<__nv_fp8_e4m3 *>(s_dynamic_buffer);
    const int s_q_size_aligned = (BLOCK_M * stride_q + 15) & ~15;

    __nv_fp8_e4m3 *s_k = s_q + s_q_size_aligned;
    const int s_k_size_aligned = (BLOCK_N * stride_k + 15) & ~15;

    __nv_fp8_e4m3 *s_v = s_k + s_k_size_aligned;
    const int s_v_size_aligned = (HEAD_DIM * stride_v + 15) & ~15;

    auto s_row_m = reinterpret_cast<float *>(s_v + s_v_size_aligned);
    float *s_row_l = s_row_m + BLOCK_M;
    auto s_acc_o = reinterpret_cast<int *>(s_row_l + BLOCK_M);
    const int s_acc_o_size_aligned = (BLOCK_M * HEAD_DIM + 3) & ~3;

    int *s_tile_scores = s_acc_o + s_acc_o_size_aligned;
    constexpr int s_tile_scores_size_aligned = (BLOCK_M * BLOCK_N + 3) & ~3;
    auto s_tile_scores_uint8 = reinterpret_cast<uint8_t *>(s_tile_scores + s_tile_scores_size_aligned);

    for (int i = tid; i < (BLOCK_M * HEAD_DIM); i += blockDim.x) {
        s_acc_o[i] = 0;
    }
    if (tid < BLOCK_M) {
        s_row_m[tid] = -1e20f;
        s_row_l[tid] = 0.0f;
    }
    __syncthreads();

    const int start_m = tile_m_idx * BLOCK_M;
    const int q_global_stride = num_heads * HEAD_DIM;
    const int kv_global_stride = num_kv_heads * HEAD_DIM;

    const int batch_q_offset = batch_idx * q_seq_len * q_global_stride;
    const int batch_kv_offset = batch_idx * kv_seq_len * kv_global_stride;

    for (int m = warp_id; m < BLOCK_M; m += num_warps) {
        if (start_m + m < q_seq_len) {
            for (int d = lane_id * 16; d < HEAD_DIM; d += 32 * 16) {
                *reinterpret_cast<int4 *>(&s_q[m * stride_q + d]) =
                        *reinterpret_cast<const int4 *>(&query[batch_q_offset + (start_m + m) * q_global_stride + q_head_idx * HEAD_DIM + d]);
            }
        } else {
            for (int d = lane_id * 16; d < HEAD_DIM; d += 32 * 16) {
                *reinterpret_cast<int4 *>(&s_q[m * stride_q + d]) = make_int4(0, 0, 0, 0);
            }
        }
    }
    __syncthreads();

    const int num_n_tiles = (kv_seq_len + BLOCK_N - 1) / BLOCK_N;

    for (int tile_n_idx = 0; tile_n_idx < num_n_tiles; ++tile_n_idx) {
        const int start_n = tile_n_idx * BLOCK_N;

        for (int n = warp_id; n < BLOCK_N; n += num_warps) {
            if (start_n + n < kv_seq_len) {
                for (int d = lane_id * 16; d < HEAD_DIM; d += 32 * 16) {
                    *reinterpret_cast<int4 *>(&s_k[n * stride_k + d]) =
                            *reinterpret_cast<const int4 *>(&key[batch_kv_offset + (start_n + n) * kv_global_stride + kv_head_idx * HEAD_DIM + d]);
                }
            } else {
                for (int d = lane_id * 16; d < HEAD_DIM; d += 32 * 16) {
                    *reinterpret_cast<int4 *>(&s_k[n * stride_k + d]) = make_int4(0, 0, 0, 0);
                }
            }
        }

        for (int d = warp_id; d < HEAD_DIM; d += num_warps) {
            for (int n = lane_id * 16; n < BLOCK_N; n += 32 * 16) {
                if (start_n + n < kv_seq_len) {
                    int4 reg_v;
                    auto *bytes_v = reinterpret_cast<uint8_t *>(&reg_v);
#pragma unroll
                    for (int b = 0; b < 16; ++b) {
                        bytes_v[b] = value[batch_kv_offset + (start_n + n + b) * kv_global_stride + kv_head_idx * HEAD_DIM + d].__x;
                    }
                    *reinterpret_cast<int4 *>(&s_v[d * stride_v + n]) = reg_v;
                } else {
                    *reinterpret_cast<int4 *>(&s_v[d * stride_v + n]) = make_int4(0, 0, 0, 0);
                }
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
                    const auto q_ptr = reinterpret_cast<const unsigned char *>(s_q + wm * stride_q + dk);
                    const auto k_ptr = reinterpret_cast<const unsigned char *>(s_k + wn * stride_k + dk);

                    load_matrix_sync(q_frag, q_ptr, stride_q);
                    load_matrix_sync(k_frag, k_ptr, stride_k);
                    mma_sync(c_frag, q_frag, k_frag, c_frag);
                }
                store_matrix_sync(s_tile_scores + wm * BLOCK_N + wn, c_frag, BLOCK_N, mem_row_major);
            }
        }
        __syncthreads();

        for (int wm = warp_id * WMMA_M; wm < BLOCK_M; wm += num_warps * WMMA_M) {
            for (int m_offset = 0; m_offset < WMMA_M; ++m_offset) {
                int m = wm + m_offset;
                if (start_m + m >= q_seq_len) continue;

                float local_tile_max = -1e20f;
#pragma unroll
                for (int k = 0; k < 2; ++k) {
                    int n = lane_id * 2 + k;
                    if (start_n + n < kv_seq_len && start_m + m >= start_n + n) {
                        float score = static_cast<float>(s_tile_scores[m * BLOCK_N + n]) * scale;
                        local_tile_max = fmaxf(local_tile_max, score);
                    }
                }

                local_tile_max = warp_reduce_max(local_tile_max);

                const float old_m = s_row_m[m];
                const float new_m = fmaxf(old_m, local_tile_max);
                const float scale_old = expf(old_m - new_m);

                float p_sum = 0.0f;
#pragma unroll
                for (int k = 0; k < 2; ++k) {
                    int n = lane_id * 2 + k;
                    float score = -1e20f;
                    if (start_n + n < kv_seq_len && start_m + m >= start_n + n) {
                        score = static_cast<float>(s_tile_scores[m * BLOCK_N + n]) * scale;
                    }

                    float s_exp = score > -1e19f ? expf(score - new_m) : 0.0f;
                    p_sum += s_exp;
                    s_tile_scores_uint8[m * BLOCK_N + n] = static_cast<uint8_t>(s_exp * 255.0f);
                }

                p_sum = warp_reduce_sum(p_sum);

                if (lane_id == 0) {
                    s_row_l[m] = s_row_l[m] * scale_old + p_sum;
                    s_row_m[m] = new_m;
                }

                for (int d = lane_id; d < HEAD_DIM; d += 32) {
                    s_acc_o[m * HEAD_DIM + d] = __float2int_rn(static_cast<float>(s_acc_o[m * HEAD_DIM + d]) * scale_old);
                }
                __syncwarp();
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
                    const auto s_tile_scores_ptr = reinterpret_cast<const unsigned char *>(s_tile_scores_uint8 + wm * BLOCK_N + dk);
                    const auto s_v_ptr = reinterpret_cast<const unsigned char *>(s_v + wd + dk * stride_v);

                    load_matrix_sync(s_frag, s_tile_scores_ptr, BLOCK_N);
                    load_matrix_sync(v_frag, s_v_ptr, stride_v);
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
                const float final_float_val = static_cast<float>(s_acc_o[m * HEAD_DIM + d]) * inv_l * (1.0f / 255.0f);
                output[batch_q_offset + (start_m + m) * q_global_stride + q_head_idx * HEAD_DIM + d] = __nv_fp8_e4m3(final_float_val);
            }
        }
    }
}
