#include <cuda_runtime.h>
#include <device_launch_parameters.h>

union Vector4 {
    float4 v4;
    float arr[4];
};

__global__ void adamw_vectorized_max_speed_kernel(
    float4 * __restrict__ weights,
    float4 * __restrict__ gradients,
    float4 * __restrict__ m_buffer,
    float4 * __restrict__ v_buffer,
    const int size_v4,
    const float lr,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float weight_decay,
    const float inv_bias_correction1,
    const float inv_bias_correction2
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size_v4) {
        Vector4 w_reg, g_reg, m_reg, v_reg;

        w_reg.v4 = __ldcs(&weights[idx]);
        g_reg.v4 = __ldcs(&gradients[idx]);
        m_reg.v4 = __ldcs(&m_buffer[idx]);
        v_reg.v4 = __ldcs(&v_buffer[idx]);

        const float one_minus_beta1 = 1.0f - beta1;
        const float one_minus_beta2 = 1.0f - beta2;

#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float g_val = g_reg.arr[i];
            const float w_val = w_reg.arr[i];
            float m_val = m_reg.arr[i];
            float v_val = v_reg.arr[i];

            m_val = beta1 * m_val + one_minus_beta1 * g_val;
            v_val = beta2 * v_val + one_minus_beta2 * g_val * g_val;

            m_reg.arr[i] = m_val;
            v_reg.arr[i] = v_val;

            const float m_hat = m_val * inv_bias_correction1;
            const float v_hat = v_val * inv_bias_correction2;

            const float denom = m_hat / (sqrtf(v_hat) + epsilon);

            w_reg.arr[i] = w_val - lr * (denom + weight_decay * w_val);
        }

        __stcs(&m_buffer[idx], m_reg.v4);
        __stcs(&v_buffer[idx], v_reg.v4);
        __stcs(&weights[idx], w_reg.v4);

        Vector4 zero_g;
        zero_g.v4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        __stcs(&gradients[idx], zero_g.v4);
    }
}

__global__ void adamw_scalar_max_speed_kernel(
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
    const float inv_bias_correction1,
    const float inv_bias_correction2
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        const float g = __ldcs(&gradients[idx]);
        const float w = __ldcs(&weights[idx]);
        float m = __ldcs(&m_buffer[idx]);
        float v = __ldcs(&v_buffer[idx]);

        const float one_minus_beta1 = 1.0f - beta1;
        const float one_minus_beta2 = 1.0f - beta2;

        m = beta1 * m + one_minus_beta1 * g;
        v = beta2 * v + one_minus_beta2 * g * g;

        const float m_hat = m * inv_bias_correction1;
        const float v_hat = v * inv_bias_correction2;

        const float denom = m_hat / (sqrtf(v_hat) + epsilon);

        __stcs(&m_buffer[idx], m);
        __stcs(&v_buffer[idx], v);
        __stcs(&weights[idx], w - lr * (denom + weight_decay * w));
        __stcs(&gradients[idx], 0.0f);
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
    const auto stream = static_cast<cudaStream_t>(stream_ptr);

    const float bias_correction1 = 1.0f - powf(beta1, step);
    const float bias_correction2 = 1.0f - powf(beta2, step);

    const float inv_bias_correction1 = bias_correction1 > 0.0f ? 1.0f / bias_correction1 : 1.0f;
    const float inv_bias_correction2 = bias_correction2 > 0.0f ? 1.0f / bias_correction2 : 1.0f;

    constexpr int threads = 256;

    const bool is_aligned = reinterpret_cast<uintptr_t>(weights) % 16 == 0 &&
                            reinterpret_cast<uintptr_t>(gradients) % 16 == 0 &&
                            reinterpret_cast<uintptr_t>(m_buffer) % 16 == 0 &&
                            reinterpret_cast<uintptr_t>(v_buffer) % 16 == 0;

    if (size % 4 == 0 && is_aligned) {
        const int size_v4 = size / 4;
        const int blocks = (size_v4 + threads - 1) / threads;

        adamw_vectorized_max_speed_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<float4 *>(weights),
            reinterpret_cast<float4 *>(gradients),
            reinterpret_cast<float4 *>(m_buffer),
            reinterpret_cast<float4 *>(v_buffer),
            size_v4, lr, beta1, beta2, epsilon, weight_decay,
            inv_bias_correction1, inv_bias_correction2
        );
    } else {
        const int blocks = (size + threads - 1) / threads;
        adamw_scalar_max_speed_kernel<<<blocks, threads, 0, stream>>>(
            weights, gradients, m_buffer, v_buffer,
            size, lr, beta1, beta2, epsilon, weight_decay,
            inv_bias_correction1, inv_bias_correction2
        );
    }
}
}
