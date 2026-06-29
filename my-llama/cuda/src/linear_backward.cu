#include <cuda_runtime.h>

__global__ void matmul_backward_weights_kernel(float *const d_weights,
                                               const float *const input,
                                               const float *const d_output,
                                               const int batch_size,
                                               const int out_features,
                                               const int in_features) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < in_features && col < out_features) {
        float sum = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            sum += input[b * in_features + row] * d_output[b * out_features + col];
        }
        d_weights[row * out_features + col] += sum;
    }
}

__global__ void matmul_backward_input_kernel(float *const d_input,
                                             const float *const d_output,
                                             const float *const weights,
                                             const int batch_size,
                                             const int out_features,
                                             const int in_features) {
    const int b = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < batch_size && col < in_features) {
        float sum = 0.0f;
        for (int o = 0; o < out_features; ++o) {
            sum += d_output[b * out_features + o] * weights[col * out_features + o];
        }
        d_input[b * in_features + col] = sum;
    }
}

extern "C" {
__declspec(dllexport) void launch_matmul_backward_weights(float *d_weights,
                                                          const float *input,
                                                          const float *d_output,
                                                          int batch_size,
                                                          int out_features,
                                                          int in_features) {
    dim3 threads(16, 16);
    dim3 blocks((out_features + threads.x - 1) / threads.x, (in_features + threads.y - 1) / threads.y);

    matmul_backward_weights_kernel<<<blocks, threads>>>(d_weights, input, d_output, batch_size, out_features, in_features);
}

__declspec(dllexport) void launch_matmul_backward_input(float *d_input,
                                                        const float *d_output,
                                                        const float *weights,
                                                        int batch_size,
                                                        int out_features,
                                                        int in_features) {
    dim3 threads(16, 16);
    dim3 blocks((in_features + threads.x - 1) / threads.x, (batch_size + threads.y - 1) / threads.y);

    matmul_backward_input_kernel<<<blocks, threads>>>(d_input, d_output, weights, batch_size, out_features, in_features);
}
}
