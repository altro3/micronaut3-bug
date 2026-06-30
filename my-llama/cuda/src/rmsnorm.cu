#include <cuda_runtime.h>

__global__ void rms_norm_kernel(
    float * __restrict__ output,
    const float * __restrict__ input,
    const float * __restrict__ weight,
    const int hidden_size,
    const float epsilon
) {
    const int row_idx = blockIdx.x;
    const float *const x = input + row_idx * hidden_size;
    float *const y = output + row_idx * hidden_size;

    extern __shared__ float s_data[];
    const int tid = threadIdx.x;

    float sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        sum += x[i] * x[i];
    }

    s_data[tid] = (tid < hidden_size) ? sum : 0.0f;
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

extern "C" {
void launch_rms_norm(
    float *output,
    const float *input,
    const float *weight,
    const int batch_size,
    const int hidden_size,
    const float epsilon,
    void *stream_ptr
) {
    const int blocks = batch_size;
    const int threads = hidden_size < 256 ? (hidden_size + 31) / 32 * 32 : 256;
    const size_t shared_mem_size = threads * sizeof(float);
    const auto stream = static_cast<cudaStream_t>(stream_ptr);

    rms_norm_kernel<<<blocks, threads, shared_mem_size, stream>>>(
        output, input, weight, hidden_size, epsilon
    );
}
}
