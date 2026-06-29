#ifndef KERNELS_H
#define KERNELS_H

extern "C" {
// Отладочный кернел
void test_cuda_setup(float *d_array, int size);

// Ядро слоя нормализации
void launch_rms_norm(float *output,
                     const float *input,
                     const float *weight,
                     int batch_size,
                     int hidden_size,
                     float epsilon);

// НОВЫЕ: Ядра для слоя SwiGLU
void launch_matmul(float *C,
                   const float *A,
                   const float *B,
                   int M, int N, int K);

void launch_swish_glu(float *output,
                      const float *gate_input,
                      const float *up_input,
                      int size);
}

#endif
