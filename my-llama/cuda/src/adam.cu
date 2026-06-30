#include <cuda_runtime.h>

__global__ void adamw_kernel(
    float * __restrict__ weights,
    float * __restrict__ gradients,
    float * __restrict__ m_buffer,
    float * __restrict__ v_buffer,
    const int size,
    const float lr,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float weight_decay,
    const float step
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        const float g = gradients[idx];
        const float w = weights[idx];

        const float m = beta1 * m_buffer[idx] + (1.0f - beta1) * g;
        const float v = beta2 * v_buffer[idx] + (1.0f - beta2) * g * g;

        m_buffer[idx] = m;
        v_buffer[idx] = v;

        const float bias_correction1 = 1.0f - powf(beta1, step);
        const float bias_correction2 = 1.0f - powf(beta2, step);

        const float m_hat = m / bias_correction1;
        const float v_hat = v / bias_correction2;

        weights[idx] = w - lr * (m_hat / (sqrtf(v_hat) + epsilon) + weight_decay * w);

        gradients[idx] = 0.0f;
    }
}

extern "C" {
void launch_adamw(
    float *weights,
    float *gradients,
    float *m_buffer,
    float *v_buffer,
    const int size,
    const float lr,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float weight_decay,
    const float step,
    void *stream_ptr
) {
    constexpr int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    const auto stream = static_cast<cudaStream_t>(stream_ptr);

    adamw_kernel<<<blocks, threads, 0, stream>>>(
        weights, gradients, m_buffer, v_buffer, size, lr, beta1, beta2, epsilon, weight_decay, step
    );
}
}
