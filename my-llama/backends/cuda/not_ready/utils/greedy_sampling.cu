#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdint.h>

__global__ void greedy_sampling_kernel(
    int * __restrict__ token_id,
    const float * __restrict__ logits,
    const int vocab_size
) {
    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int num_warps = (blockDim.x + 31) / 32;

    float max_val = -1e20f;
    int max_idx = 0;

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        const float lgt = logits[i];
        if (lgt > max_val) {
            max_val = lgt;
            max_idx = i;
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float remote_max = __shfl_xor_sync(0xFFFFFFFF, max_val, offset);
        const int remote_idx = __shfl_xor_sync(0xFFFFFFFF, max_idx, offset);
        if (remote_max > max_val) {
            max_val = remote_max;
            max_idx = remote_idx;
        }
    }

    extern __shared__ uint8_t s_dynamic_mem[];
    const auto s_warp_maxes = reinterpret_cast<float *>(s_dynamic_mem);
    const auto s_warp_indices = reinterpret_cast<int *>(s_warp_maxes + (num_warps + 3 & ~3));

    if (lane_id == 0) {
        s_warp_maxes[warp_id] = max_val;
        s_warp_indices[warp_id] = max_idx;
    }
    __syncthreads();

    if (tid == 0) {
        float block_max = -1e20f;
        int block_max_idx = 0;

        for (int w = 0; w < num_warps; ++w) {
            if (s_warp_maxes[w] > block_max) {
                block_max = s_warp_maxes[w];
                block_max_idx = s_warp_indices[w];
            }
        }
        *token_id = block_max_idx;
    }
}

extern "C" {
void launch_greedy_sampling(
    int *token_id,
    const float *logits,
    const int vocab_size,
    const int threads_per_block,
    void *stream_ptr
) {
    if (vocab_size <= 0 || threads_per_block <= 0) return;

    const auto stream = static_cast<cudaStream_t>(stream_ptr);
    const int num_warps = (threads_per_block + 31) / 32;
    const int num_warps_aligned = num_warps + 3 & ~3;
    const int shared_mem_size = num_warps_aligned * (sizeof(float) + sizeof(int));

    greedy_sampling_kernel<<<1, threads_per_block, shared_mem_size, stream>>>(
        token_id, logits, vocab_size
    );
}
}
