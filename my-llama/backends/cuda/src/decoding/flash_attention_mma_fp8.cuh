#ifndef FLASH_ATTENTION_MMA_FP8_CUH
#define FLASH_ATTENTION_MMA_FP8_CUH

#include "../flash_attention_types.cuh"

__device__ __forceinline__ void mma_m16n8k32_fp8(
    float *d,
    const uint32_t *a,
    const uint32_t *b,
    const float *c
) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b[0]), "r"(b[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
}

template<int REG_COUNT_M, int REG_COUNT_N>
struct RegisterAccumulator {
    float data[REG_COUNT_M][REG_COUNT_N][4];

    __device__ __forceinline__ void clear() {
#pragma unroll
        for (int i = 0; i < REG_COUNT_M; ++i) {
#pragma unroll
            for (int j = 0; j < REG_COUNT_N; ++j) {
                data[i][j][0] = 0.0f;
                data[i][j][1] = 0.0f;
                data[i][j][2] = 0.0f;
                data[i][j][3] = 0.0f;
            }
        }
    }

    __device__ __forceinline__ void scale_elements(float alpha) {
#pragma unroll
        for (int i = 0; i < REG_COUNT_M; ++i) {
#pragma unroll
            for (int j = 0; j < REG_COUNT_N; ++j) {
                data[i][j][0] *= alpha;
                data[i][j][1] *= alpha;
                data[i][j][2] *= alpha;
                data[i][j] *= alpha;
            }
        }
    }
};

template<int HEAD_DIM>
__device__ __forceinline__ void compute_qk_t_block(
    RegisterAccumulator<BLOCK_M / MMA_M, BLOCK_N / MMA_N> &accum_scores,
    const __nv_fp8_e4m3 *s_q,
    const __nv_fp8_e4m3 *s_k,
    int lane_id,
    int stride_q,
    int stride_k
) {
    uint32_t reg_a[4];
    uint32_t reg_b[2];

    const int mma_thread_m = lane_id % 16;
    const int mma_thread_n = lane_id / 16;

#pragma unroll
    for (int dk = 0; dk < HEAD_DIM; dk += MMA_K) {
#pragma unroll
        for (int i = 0; i < BLOCK_M / MMA_M; ++i) {
            const int logical_m = i * MMA_M + mma_thread_m;

            const auto *smem_ptr_a = reinterpret_cast<const uint32_t *>(
                s_q + logical_m * stride_q + dk + (lane_id % 4) * 4
            );
            reg_a[0] = smem_ptr_a[0];
            reg_a[1] = smem_ptr_a[1];
            reg_a[2] = smem_ptr_a[2];
            reg_a[3] = smem_ptr_a[3];

#pragma unroll
            for (int j = 0; j < BLOCK_N / MMA_N; ++j) {
                const int logical_n = j * MMA_N + mma_thread_n;

                const auto *smem_ptr_b = reinterpret_cast<const uint32_t *>(
                    s_k + logical_n * stride_k + dk + (lane_id / 4) * 4
                );
                reg_b[0] = smem_ptr_b[0];
                reg_b[1] = smem_ptr_b[1];

                mma_m16n8k32_fp8(
                    accum_scores.data[i][j],
                    reg_a,
                    reg_b,
                    accum_scores.data[i][j]
                );
            }
        }
    }
}

#endif
