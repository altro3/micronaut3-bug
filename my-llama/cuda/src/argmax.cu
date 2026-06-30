#include "kernels.h"
#include <cuda_runtime.h>
#include <math.h>

__global__ void argmax_kernel(int *const output_index, const float *const logits, const int vocab_size) {
    extern __shared__ float s_max_val[];
    const auto s_max_idx = reinterpret_cast<int *>(&s_max_val[blockDim.x]);
    const int tid = threadIdx.x;

    float max_val = -INFINITY;
    int max_idx = -1;

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }

    s_max_val[tid] = max_val;
    s_max_idx[tid] = max_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_max_val[tid + s] > s_max_val[tid]) {
                s_max_val[tid] = s_max_val[tid + s];
                s_max_idx[tid] = s_max_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *output_index = s_max_idx[0];
    }
}

extern "C" {
void launch_argmax(int *output_index, const float *logits, const int vocab_size) {
    constexpr int threads = 256;
    constexpr int shared_mem_size = threads * sizeof(float) + threads * sizeof(int);

    argmax_kernel<<<1, threads, shared_mem_size>>>(output_index, logits, vocab_size);
}
}
