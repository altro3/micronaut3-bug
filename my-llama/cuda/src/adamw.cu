#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" {
__global__ void adamw_scalar_fallback_kernel(
    float * __restrict__ weights,
    const float * __restrict__ gradients,
    float * __restrict__ m_buffer,
    float * __restrict__ v_buffer,
    const int size,
    const float lr,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float weight_decay,
    const float bias_corr_ratio
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        const float one_minus_beta1 = 1.0f - beta1;
        const float one_minus_beta2 = 1.0f - beta2;

        float w = weights[idx];
        const float g = gradients[idx];
        float m = m_buffer[idx];
        float v = v_buffer[idx];

        m = beta1 * m + one_minus_beta1 * g;
        v = beta2 * v + one_minus_beta2 * g * g;

        const float inv_denom = __frcp_rn(__fsqrt_rn(v) + epsilon);
        w -= lr * (m * bias_corr_ratio * inv_denom + weight_decay * w);

        m_buffer[idx] = m;
        v_buffer[idx] = v;
        weights[idx] = w;
    }
}

__global__ void __launch_bounds__(512, 2) adamw_ultimate_blackwell_kernel(
    float4 * __restrict__ weights,
    const float4 * __restrict__ gradients,
    float4 * __restrict__ m_buffer,
    float4 * __restrict__ v_buffer,
    const int size_v4,
    const int total_elements,
    const float lr,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float weight_decay,
    const float bias_corr_ratio
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const float one_minus_beta1 = 1.0f - beta1;
    const float one_minus_beta2 = 1.0f - beta2;

    if (idx < size_v4) {
        float4 w = __ldcs(&weights[idx]);
        const float4 g = __ldcs(&gradients[idx]);
        float4 m = __ldcs(&m_buffer[idx]);
        float4 v = __ldcs(&v_buffer[idx]);

        m.x = beta1 * m.x + one_minus_beta1 * g.x;
        m.y = beta1 * m.y + one_minus_beta1 * g.y;
        m.z = beta1 * m.z + one_minus_beta1 * g.z;
        m.w = beta1 * m.w + one_minus_beta1 * g.w;

        v.x = beta2 * v.x + one_minus_beta2 * (g.x * g.x);
        v.y = beta2 * v.y + one_minus_beta2 * (g.y * g.y);
        v.z = beta2 * v.z + one_minus_beta2 * (g.z * g.z);
        v.w = beta2 * v.w + one_minus_beta2 * (g.w * g.w);

        const float inv_denom_x = __frcp_rn(__fsqrt_rn(v.x) + epsilon);
        const float inv_denom_y = __frcp_rn(__fsqrt_rn(v.y) + epsilon);
        const float inv_denom_z = __frcp_rn(__fsqrt_rn(v.z) + epsilon);
        const float inv_denom_w = __frcp_rn(__fsqrt_rn(v.w) + epsilon);

        w.x -= lr * (m.x * bias_corr_ratio * inv_denom_x + weight_decay * w.x);
        w.y -= lr * (m.y * bias_corr_ratio * inv_denom_y + weight_decay * w.y);
        w.z -= lr * (m.z * bias_corr_ratio * inv_denom_z + weight_decay * w.z);
        w.w -= lr * (m.w * bias_corr_ratio * inv_denom_w + weight_decay * w.w);

        __stcs(&m_buffer[idx], m);
        __stcs(&v_buffer[idx], v);
        __stcs(&weights[idx], w);
    } else {
        const int scalar_idx = size_v4 * 4 + (idx - size_v4);
        if (scalar_idx < total_elements) {
            float *s_weights = reinterpret_cast<float *>(weights);
            const float *s_gradients = reinterpret_cast<const float *>(gradients);
            float *s_m_buffer = reinterpret_cast<float *>(m_buffer);
            float *s_v_buffer = reinterpret_cast<float *>(v_buffer);

            float w = __ldcs(&s_weights[scalar_idx]);
            const float g = __ldcs(&s_gradients[scalar_idx]);
            float m = __ldcs(&s_m_buffer[scalar_idx]);
            float v = __ldcs(&s_v_buffer[scalar_idx]);

            m = beta1 * m + one_minus_beta1 * g;
            v = beta2 * v + one_minus_beta2 * g * g;

            const float inv_denom = __frcp_rn(__fsqrt_rn(v) + epsilon);
            w -= lr * (m * bias_corr_ratio * inv_denom + weight_decay * w);

            __stcs(&s_m_buffer[scalar_idx], m);
            __stcs(&s_v_buffer[scalar_idx], v);
            __stcs(&s_weights[scalar_idx], w);
        }
    }
}

extern "C" void launch_adamw(
    float *weights, const float *gradients, float *m_buffer, float *v_buffer,
    const int size, const float lr, const float beta1, const float beta2,
    const float epsilon, const float weight_decay, const float step,
    void *stream_ptr
) {
    const auto stream = static_cast<cudaStream_t>(stream_ptr);

    const float bias_correction1 = 1.0f - powf(beta1, step);
    const float bias_correction2 = 1.0f - powf(beta2, step);

    const float inv_bias_correction1 = bias_correction1 > 0.0f ? 1.0f / bias_correction1 : 1.0f;
    const float bias_corr_ratio = inv_bias_correction1 * sqrtf(bias_correction2);

    constexpr int threads = 256;

    const bool is_aligned = reinterpret_cast<uintptr_t>(weights) % 16 == 0
                            && reinterpret_cast<uintptr_t>(gradients) % 16 == 0
                            && reinterpret_cast<uintptr_t>(m_buffer) % 16 == 0
                            && reinterpret_cast<uintptr_t>(v_buffer) % 16 == 0;

    if (is_aligned) [[likely]] {
        const int size_v4 = size / 4;
        const int total_threads_needed = (size % 4 == 0) ? size_v4 : size_v4 + 1;
        const int blocks = (total_threads_needed + threads - 1) / threads;

        adamw_ultimate_blackwell_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<float4 *>(weights),
            reinterpret_cast<const float4 *>(gradients),
            reinterpret_cast<float4 *>(m_buffer),
            reinterpret_cast<float4 *>(v_buffer),
            size_v4, size, lr, beta1, beta2, epsilon, weight_decay, bias_corr_ratio
        );
    } else [[unlikely]] {
        const int blocks = (size + threads - 1) / threads;
        adamw_scalar_fallback_kernel<<<blocks, threads, 0, stream>>>(
            weights, gradients, m_buffer, v_buffer,
            size, lr, beta1, beta2, epsilon, weight_decay, bias_corr_ratio
        );
    }
}
}
