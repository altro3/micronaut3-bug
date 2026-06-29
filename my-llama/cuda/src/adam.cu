#include <cuda_runtime.h>

// Параллельный кернел AdamW: один поток обновляет один параметр модели
__global__ void adamw_kernel(float *const weights,
                             float *const gradients,
                             float *const m_buffer,
                             float *const v_buffer,
                             const int size,
                             const float lr,
                             const float beta1,
                             const float beta2,
                             const float epsilon,
                             const float weight_decay,
                             const float step) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        const float g = gradients[idx];
        const float w = weights[idx];

        // 1. Обновляем первый и второй моменты (скользящее среднее)
        const float m = beta1 * m_buffer[idx] + (1.0f - beta1) * g;
        const float v = beta2 * v_buffer[idx] + (1.0f - beta2) * g * g;

        m_buffer[idx] = m;
        v_buffer[idx] = v;

        // 2. Коррекция смещения (Bias Correction)
        const float bias_correction1 = 1.0f - powf(beta1, step);
        const float bias_correction2 = 1.0f - powf(beta2, step);

        const float m_hat = m / bias_correction1;
        const float v_hat = v / bias_correction2;

        // 3. Обновление веса с учетом затухания (Weight Decay)
        weights[idx] = w - lr * (m_hat / (sqrtf(v_hat) + epsilon) + weight_decay * w);

        // 4. Зануляем градиент для следующей итерации обучения (каноничный шаг)
        gradients[idx] = 0.0f;
    }
}

extern "C" {
// Явный экспорт для Windows / MSVC линкера
__declspec(dllexport) void launch_adamw(float *weights,
                                        float *gradients,
                                        float *m_buffer,
                                        float *v_buffer,
                                        const int size,
                                        const float lr,
                                        const float beta1,
                                        const float beta2,
                                        const float epsilon,
                                        const float weight_decay,
                                        const float step) {
    constexpr int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    adamw_kernel<<<blocks, threads>>>(
        weights, gradients, m_buffer, v_buffer, size, lr, beta1, beta2, epsilon, weight_decay, step
    );
}
}
