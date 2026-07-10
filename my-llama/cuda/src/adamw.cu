#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" {
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
        float4 w_v4 = __ldcs(&weights[idx]);
        const float4 g_v4 = __ldcs(&gradients[idx]);
        float4 m_v4 = __ldcs(&m_buffer[idx]);
        float4 v_v4 = __ldcs(&v_buffer[idx]);

        const float one_minus_beta1 = 1.0f - beta1;
        const float one_minus_beta2 = 1.0f - beta2;

        m_v4.x = beta1 * m_v4.x + one_minus_beta1 * g_v4.x;
        v_v4.x = beta2 * v_v4.x + one_minus_beta2 * g_v4.x * g_v4.x;
        const float sqrt_v_x = __fsqrt_rn(v_v4.x * inv_bias_correction2);
        const float inv_denom_x = __frcp_rn(sqrt_v_x + epsilon);
        const float denom_x = (m_v4.x * inv_bias_correction1) * inv_denom_x;
        w_v4.x = w_v4.x - lr * (denom_x + weight_decay * w_v4.x);

        m_v4.y = beta1 * m_v4.y + one_minus_beta1 * g_v4.y;
        v_v4.y = beta2 * v_v4.y + one_minus_beta2 * g_v4.y * g_v4.y;
        const float sqrt_v_y = __fsqrt_rn(v_v4.y * inv_bias_correction2);
        const float inv_denom_y = __frcp_rn(sqrt_v_y + epsilon);
        const float denom_y = (m_v4.y * inv_bias_correction1) * inv_denom_y;
        w_v4.y = w_v4.y - lr * (denom_y + weight_decay * w_v4.y);

        m_v4.z = beta1 * m_v4.z + one_minus_beta1 * g_v4.z;
        v_v4.z = beta2 * v_v4.z + one_minus_beta2 * g_v4.z * g_v4.z;
        const float sqrt_v_z = __fsqrt_rn(v_v4.z * inv_bias_correction2);
        const float inv_denom_z = __frcp_rn(sqrt_v_z + epsilon);
        const float denom_z = (m_v4.z * inv_bias_correction1) * inv_denom_z;
        w_v4.z = w_v4.z - lr * (denom_z + weight_decay * w_v4.z);

        m_v4.w = beta1 * m_v4.w + one_minus_beta1 * g_v4.w;
        v_v4.w = beta2 * v_v4.w + one_minus_beta2 * g_v4.w * g_v4.w;
        const float sqrt_v_w = __fsqrt_rn(v_v4.w * inv_bias_correction2);
        const float inv_denom_w = __frcp_rn(sqrt_v_w + epsilon);
        const float denom_w = (m_v4.w * inv_bias_correction1) * inv_denom_w;
        w_v4.w = w_v4.w - lr * (denom_w + weight_decay * w_v4.w);

        __stcs(&m_buffer[idx], m_v4);
        __stcs(&v_buffer[idx], v_v4);
        __stcs(&weights[idx], w_v4);

        __stcs(&gradients[idx], make_float4(0.0f, 0.0f, 0.0f, 0.0f));
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

        m = beta1 * m + (1.0f - beta1) * g;
        v = beta2 * v + (1.0f - beta2) * g * g;

        const float sqrt_v = __fsqrt_rn(v * inv_bias_correction2);
        const float inv_denom = __frcp_rn(sqrt_v + epsilon);
        const float denom = (m * inv_bias_correction1) * inv_denom;

        __stcs(&m_buffer[idx], m);
        __stcs(&v_buffer[idx], v);
        __stcs(&weights[idx], w - lr * (denom + weight_decay * w));
        __stcs(&gradients[idx], 0.0f);
    }
}

extern "C" void launch_adamw(
    float *weights, float *gradients, float *m_buffer, float *v_buffer,
    const int size, const float lr, const float beta1, const float beta2,
    const float epsilon, const float weight_decay, const float step,
    void *stream_ptr
) {
    const auto stream = static_cast<cudaStream_t>(stream_ptr);

    const float bias_correction1 = 1.0f - powf(beta1, step);
    const float bias_correction2 = 1.0f - powf(beta2, step);
    const float inv_bias_correction1 = bias_correction1 > 0.0f ? 1.0f / bias_correction1 : 1.0f;
    const float inv_bias_correction2 = bias_correction2 > 0.0f ? 1.0f / bias_correction2 : 1.0f;

    constexpr int threads = 256;

    const bool is_aligned = reinterpret_cast<uintptr_t>(weights) % 16 == 0
                            && reinterpret_cast<uintptr_t>(gradients) % 16 == 0
                            && reinterpret_cast<uintptr_t>(m_buffer) % 16 == 0
                            && reinterpret_cast<uintptr_t>(v_buffer) % 16 == 0;

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
