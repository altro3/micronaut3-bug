#include <cuda_runtime.h>
#include <math.h>

__global__ void softmax_attention_kernel(float * __restrict__ scores, const int current_seq_len) {
    const int head_idx = blockIdx.x;
    float *const row = scores + head_idx * current_seq_len;
    const int tid = threadIdx.x;

    extern __shared__ float s_mem[];
    float local_max = -INFINITY;

    for (int i = tid; i < current_seq_len; i += blockDim.x) {
        if (row[i] > local_max) {
            local_max = row[i];
        }
    }
    s_mem[tid] = tid < current_seq_len ? local_max : -INFINITY;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_mem[tid + s] > s_mem[tid]) {
                s_mem[tid] = s_mem[tid + s];
            }
        }
        __syncthreads();
    }
    const float row_max = s_mem[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < current_seq_len; i += blockDim.x) {
        local_sum += expf(row[i] - row_max);
    }
    s_mem[tid] = tid < current_seq_len ? local_sum : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_mem[tid] += s_mem[tid + s];
        }
        __syncthreads();
    }
    const float sum_total = s_mem[0];

    for (int i = tid; i < current_seq_len; i += blockDim.x) {
        row[i] = expf(row[i] - row_max) / sum_total;
    }
}

extern "C" {
void launch_softmax_attention(float *scores, int num_heads, const int current_seq_len, void *stream_ptr) {
    const int threads = current_seq_len < 256 ? (current_seq_len + 31) / 32 * 32 : 256;
    const size_t shared_mem_size = threads * sizeof(float);

    const auto stream = static_cast<cudaStream_t>(stream_ptr);

    softmax_attention_kernel<<<num_heads, threads, shared_mem_size, stream>>>(scores, current_seq_len);
}
}
