#include "kernels.h"
#include <cuda_runtime.h>

// Это кернел — функция, которая будет параллельно выполняться прямо на чипе GPU
__global__ void set_ones_kernel(float *d_array, const int size) {
    // Вычисляем глобальный уникальный индекс потока (Thread ID)
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Защита от выхода за границы массива
    if (idx < size) {
        d_array[idx] = 1.0f; // Видеокарта пишет единицу в ячейку памяти
    }
}

// Обертка на чистом Си, которую без проблем вызовет наш Rust
extern "C" {
void test_cuda_setup(float *d_array, const int size) {
    int threads_per_block = 256;
    // Считаем, сколько блоков потоков нужно запустить, чтобы обработать весь массив
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    // Запуск параллельного кернела на видеокарте с помощью специального синтаксиса <<< >>>
    set_ones_kernel<<<blocks_per_grid, threads_per_block>>>(d_array, size);

    // Барьер синхронизации: заставляем процессор подождать, пока GPU закончит всю работу
    cudaDeviceSynchronize();
}
}
