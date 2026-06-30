#include "kernels.h"
#include <cuda_runtime.h>
#include <math.h>

// Ядро вычисляет Safe Softmax, Cross-Entropy Loss и градиенты dL/dz = p_i - y_i
__global__ void fused_cross_entropy_kernel(
    const float * __restrict__ logits, // [num_tokens, vocab_size]
    const int * __restrict__ targets, // [num_tokens]
    float * __restrict__ d_logits, // [num_tokens, vocab_size] - выходной градиент
    float * __restrict__ losses, // [num_tokens] - лосс для каждого токена
    int vocab_size
) {
    int token_idx = blockIdx.x; // Один блок обрабатывает один токен
    int tid = threadIdx.x;

    const float *token_logits = logits + token_idx * vocab_size;
    float *token_d_logits = d_logits + token_idx * vocab_size;
    int target_label = targets[token_idx];

    // Шаг 1: Поиск максимума строки (Max Reduction) для стабильности экспонент
    float local_max = -INFINITY;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        if (token_logits[i] > local_max) {
            local_max = token_logits[i];
        }
    }

    extern __shared__ float s_mem[];
    s_mem[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            if (s_mem[tid + stride] > s_mem[tid]) {
                s_mem[tid] = s_mem[tid + stride];
            }
        }
        __syncthreads();
    }
    float global_max = s_mem[0];
    __syncthreads();

    // Шаг 2: Вычисление суммы экспонент (Sum Reduction)
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        local_sum += expf(token_logits[i] - global_max);
    }

    s_mem[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_mem[tid] += s_mem[tid + stride];
        }
        __syncthreads();
    }
    float global_sum = s_mem[0];
    __syncthreads();

    // Шаг 3: Расчет лосса и градиентов (p_i - y_i)
    if (tid == 0) {
        // L = -ln(p_target) = -((logits[target] - max) - ln(sum))
        float log_p_target = (token_logits[target_label] - global_max) - logf(global_sum);
        losses[token_idx] = -log_p_target;
    }

    // Расчет вероятностей и градиента для каждого токена словаря
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float p_i = expf(token_logits[i] - global_max) / global_sum;
        float y_i = (i == target_label) ? 1.0f : 0.0f;

        // dL/dzi = p_i - y_i
        token_d_logits[i] = p_i - y_i;
    }
}

extern "C" {
void launch_fused_cross_entropy(
    const float *logits,
    const int *targets,
    float *d_logits,
    float *losses,
    int num_tokens,
    int vocab_size
) {
    constexpr int threads = 256;
    int blocks = num_tokens;
    size_t shared_mem_size = threads * sizeof(float);

    fused_cross_entropy_kernel<<<blocks, threads, shared_mem_size>>>(
        logits, targets, d_logits, losses, vocab_size
    );
}
}
