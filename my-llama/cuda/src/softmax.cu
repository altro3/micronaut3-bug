#include "kernels.h"
#include <cuda_runtime.h>

__global__ void rms_norm_kernel(float *const output,
                                const float *const input,
                                const float *const weight,
                                const int hidden_size,
                                const float epsilon) {
    const int row_idx = blockIdx.x;

    const float *const x = input + row_idx * hidden_size;
    float *const y = output + row_idx * hidden_size;

    extern __shared__ float s_data[];
    const int tid = threadIdx.x;

    float sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        sum += x[i] * x[i];
    }
    s_data[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    float rms_inv = 0.0f;

    if (tid == 0) {
        rms_inv = 1.0f / sqrtf(s_data[0] / hidden_size + epsilon);
    }

    rms_inv = __shfl_sync(0xFFFFFFFF, rms_inv, 0);

    if (blockDim.x > 32) {
        if (tid == 0) {
            s_data[0] = rms_inv;
        }
        __syncthreads();
        rms_inv = s_data[0];
    }

    for (int i = tid; i < hidden_size; i += blockDim.x) {
        y[i] = x[i] * rms_inv * weight[i];
    }
}

__global__ void argmax_kernel(int *const output_index, const float *const logits, const int vocab_size) {
    extern __shared__ float s_max_val[];
    const auto s_max_idx = reinterpret_cast<int *>(&s_max_val[blockDim.x]);

    const int tid = threadIdx.x;

    float max_val = -INFINITY;
    int max_idx = -1;

    // Каждый поток ищет максимум в своей порции словаря
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
void launch_rms_norm(float *output,
                     const float *input,
                     const float *weight,
                     const int batch_size,
                     const int hidden_size,
                     const float epsilon) {
    const int blocks = batch_size;
    const int threads = hidden_size < 256 ? hidden_size : 256;
    const int shared_mem_size = threads * sizeof(float);

    rms_norm_kernel<<<blocks, threads, shared_mem_size>>>(output, input, weight, hidden_size, epsilon);
}

void launch_argmax(int *output_index, const float *logits, const int vocab_size) {
    constexpr int threads = 256;
    constexpr int shared_mem_size = threads * sizeof(float) + threads * sizeof(int);

    argmax_kernel<<<1, threads, shared_mem_size>>>(output_index, logits, vocab_size);
}
}
