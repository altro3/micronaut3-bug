#include <cuda_runtime.h>

// Кернел поэлементного перемножения Swish(Gate) * Up
__global__ void swish_glu_fused_kernel(float *const output,
                                       const float *const gate_input,
                                       const float *const up_input,
                                       const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        const float g = gate_input[idx];
        const float u = up_input[idx];
        const float swish = g / (1.0f + expf(-g));
        output[idx] = swish * u;
    }
}

// Параллельный кернел матричного умножения
__global__ void matmul_kernel(float *const matrix_c,
                              const float *const matrix_a,
                              const float *const matrix_b,
                              const int batch_size, const int out_features, const int in_features) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        for (int i = 0; i < in_features; ++i) {
            sum += matrix_a[row * in_features + i] * matrix_b[i * out_features + col];
        }
        matrix_c[row * out_features + col] = sum;
    }
}

// Кернел копирования текущих KV-векторов в глобальный статический кэш
__global__ void update_kv_cache_kernel(float *const k_cache,
                                       float *const v_cache,
                                       const float *const new_k,
                                       const float *const new_v,
                                       const int token_index,
                                       const int hidden_size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < hidden_size) {
        // Вычисляем глобальное смещение в кэше для текущего токена
        const int cache_offset = token_index * hidden_size + idx;

        // Переносим данные из временного локального вектора в статический буфер кэша
        k_cache[cache_offset] = new_k[idx];
        v_cache[cache_offset] = new_v[idx];
    }
}

// Кернел остаточной связи (Residual Connection): поэлементное сложение двух векторов прямо во VRAM
__global__ void residual_kernel(float *const input_output, const float *const residual_data, const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Прибавляем к основному потоку данных результат работы слоя
        input_output[idx] += residual_data[idx];
    }
}

extern "C" {
// Возвращаем const для локальных копий примитивов, чтобы порадовать статический анализатор IDEA
void launch_matmul(float *output_matrix,
                   const float *matrix_a,
                   const float *matrix_b,
                   const int batch_size, const int out_features, const int in_features) {
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((out_features + 15) / 16, (batch_size + 15) / 16);

    matmul_kernel<<<blocks_per_grid, threads_per_block>>>(output_matrix, matrix_a, matrix_b, batch_size, out_features, in_features);
}

void launch_swish_glu(float *output,
                      const float *gate_input,
                      const float *up_input,
                      const int size) {
    constexpr int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    swish_glu_fused_kernel<<<blocks, threads>>>(output, gate_input, up_input, size);
}

void launch_update_kv_cache(float *k_cache,
                            float *v_cache,
                            const float *new_k,
                            const float *new_v,
                            const int token_index,
                            const int hidden_size) {
    constexpr int threads = 256;
    const int blocks = (hidden_size + threads - 1) / threads;

    update_kv_cache_kernel<<<blocks, threads>>>(k_cache, v_cache, new_k, new_v, token_index, hidden_size);
}

// Безопасная Си-обертка для Rust
void launch_residual(float *input_output, const float *residual_data, const int size) {
    constexpr int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    residual_kernel<<<blocks, threads>>>(input_output, residual_data, size);
}
}
