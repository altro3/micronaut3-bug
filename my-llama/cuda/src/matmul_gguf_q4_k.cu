#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <stdint.h>

struct __align__(4) block_q4_K {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

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

    const int i_group = lane_id / 4;
    const int i_stripe = lane_id % 4;

    for (int block_idx = warp_id; block_idx < blocks_per_row; block_idx += num_warps) {
        const block_q4_K *block = &row_weights[block_idx];

        const float d_val = __half2float(block->d);
        const float dmin_val = __half2float(block->dmin);
        const float *block_vec_x = vec_x + block_idx * 256;

        uint8_t sc, min_sc;
        if (i_group < 4) {
            sc = block->scales[i_group * 3] & 63;
            min_sc = block->scales[i_group * 3 + 2] & 63;
        } else {
            sc = block->scales[(i_group - 4) * 3 + 1] & 63;
            min_sc = block->scales[(i_group - 4) * 3 + 2] & 63;
        }

        const float d_super = d_val * static_cast<float>(sc);
        const float m_super = dmin_val * static_cast<float>(min_sc);

        const int q_offset = i_group * 16 + i_stripe * 2;

        const uint8_t b_low0 = block->qs[q_offset];
        const uint8_t b_low1 = block->qs[q_offset + 1];
        const uint8_t b_high0 = block->qs[q_offset + 8];
        const uint8_t b_high1 = block->qs[q_offset + 9];

        const int x_idx_low = i_group * 32 + i_stripe * 4;
        const int x_idx_high = i_group * 32 + i_stripe * 4 + 16;

        const float4 vx_low = *reinterpret_cast<const float4 *>(&block_vec_x[x_idx_low]);
        const float4 vx_high = *reinterpret_cast<const float4 *>(&block_vec_x[x_idx_high]);

        thread_acc += (d_super * static_cast<float>(b_low0 & 0x0F) - m_super) * vx_low.x;
        thread_acc += (d_super * static_cast<float>(b_low0 >> 4) - m_super) * vx_low.y;
        thread_acc += (d_super * static_cast<float>(b_low1 & 0x0F) - m_super) * vx_low.z;
        thread_acc += (d_super * static_cast<float>(b_low1 >> 4) - m_super) * vx_low.w;

        thread_acc += (d_super * static_cast<float>(b_high0 & 0x0F) - m_super) * vx_high.x;
        thread_acc += (d_super * static_cast<float>(b_high0 >> 4) - m_super) * vx_high.y;
        thread_acc += (d_super * static_cast<float>(b_high1 & 0x0F) - m_super) * vx_high.z;
        thread_acc += (d_super * static_cast<float>(b_high1 >> 4) - m_super) * vx_high.w;
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
