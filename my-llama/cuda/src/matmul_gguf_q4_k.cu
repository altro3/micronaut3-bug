#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

typedef struct {
    float d;
    float dmin;
    uint8_t scales[12];
    uint8_t qs[128];
} block_q4_K;

__device__ __forceinline__ float warp_reduce_sum_matmul(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void gemv_gguf_q4_k_fused_ultra_kernel(
    float * __restrict__ output,
    const block_q4_K * __restrict__ weights,
    const float * __restrict__ vec_x,
    const int out_features,
    const int in_features
) {
    const int row = blockIdx.x;
    if (row >= out_features) return;

    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int num_warps = blockDim.x / 32;

    const int blocks_per_row = in_features / 256;
    const block_q4_K *row_weights = weights + row * blocks_per_row;

    __shared__ float s_warp_accs[32];
    float thread_acc = 0.0f;

    for (int block_idx = warp_id; block_idx < blocks_per_row; block_idx += num_warps) {
        const block_q4_K *block = &row_weights[block_idx];

        const float d = __ldcs(&block->d);
        const float dmin = __ldcs(&block->dmin);
        const float *block_vec_x = vec_x + block_idx * 256;

        const int subblock_idx = lane_id / 4;

        const int scale_offset = subblock_idx / 4 * 6;
        const int scale_group = subblock_idx % 4;

        const uint8_t scale_byte = block->scales[scale_offset + scale_group / 2];
        const uint8_t extra_byte = block->scales[scale_offset + 2];

        float sc_1, sc_2;
        if (scale_group % 2 == 0) {
            sc_1 = static_cast<float>(scale_byte & 0x0F);
            sc_2 = static_cast<float>(extra_byte & 0x0F);
        } else {
            sc_1 = static_cast<float>(scale_byte >> 4);
            sc_2 = static_cast<float>(extra_byte >> 4);
        }

        const float scale_val = sc_1 * d;
        const float min_val = sc_2 * dmin;

        const int step = lane_id % 4;

#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int x_idx = subblock_idx * 32 + step * 8 + j;

            const int q_offset = subblock_idx * 16 + step * 4 + j / 2;
            const uint8_t byte_q = block->qs[q_offset];

            float w;
            if (j % 2 == 0) {
                w = static_cast<float>(byte_q & 0x0F) * scale_val - min_val;
            } else {
                w = static_cast<float>(byte_q >> 4) * scale_val - min_val;
            }

            thread_acc += w * __ldcs(&block_vec_x[x_idx]);
        }
    }

    const float warp_sum = warp_reduce_sum_matmul(thread_acc);
    if (lane_id == 0) {
        s_warp_accs[warp_id] = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        const float val = tid < num_warps ? s_warp_accs[lane_id] : 0.0f;
        const float block_final_sum = warp_reduce_sum_matmul(val);
        if (tid == 0) {
            output[row] = block_final_sum;
        }
    }
}

extern "C" {
void launch_matmul_gguf_q4_k(
    float *output,
    const void *weights,
    const float *vec_x,
    const int out_features,
    const int in_features,
    void *stream_ptr
) {
    const auto stream = static_cast<cudaStream_t>(stream_ptr);
    constexpr int threads = 256;
    const int blocks = out_features;

    gemv_gguf_q4_k_fused_ultra_kernel<<<blocks, threads, 0, stream>>>(
        output,
        static_cast<const block_q4_K *>(weights),
        vec_x,
        out_features,
        in_features
    );
}
}
