

// 1. Кернел поэлементного перемножения Swish(Gate) * Up (Fused Activation)
// Каждый поток обрабатывает один элемент векторов
__global__ void swish_glu_fused_kernel(float *const output,
                                       const float *const gate_input,
                                       const float *const up_input,
                                       const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        const float g = gate_input[idx];
        const float u = up_input[idx];

        // Формула SiLU (Swish): g / (1.0f + expf(-g))
        const float swish = g / (1.0f + expf(-g));

        // Поэлементное умножение (Gated Linear Unit)
        output[idx] = swish * u;
    }
}

// 2. Классический параллельный кернел матричного умножения (C = A * B)
// Адаптирован под плоские непрерывные массивы f32
__global__ void matmul_kernel(float *const C,
                              const float *const A,
                              const float *const B,
                              const int M, const int N, const int K) {
    // Вычисляем строку и столбец матрицы, за которые отвечает данный поток
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

extern "C" {
// Обертка матричного умножения для Rust
void launch_matmul(float *const C, const float *const A, const float *const B,
                   const int M, const int N, const int K) {
    // Настраиваем двумерную сетку потоков для оптимальной утилизации ядер Blackwell
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((N + 15) / 16, (M + 15) / 16);

    matmul_kernel<<<blocks_per_grid, threads_per_block>>>(C, A, B, M, N, K);
}

// Обертка слияния Swish-GLU для Rust
void launch_swish_glu(float *const output, const float *const gate_input,
                      const float *const up_input, const int size) {
    constexpr int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    swish_glu_fused_kernel<<<blocks, threads>>>(output, gate_input, up_input, size);
}
}
