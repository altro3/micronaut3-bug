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

// Полноценные инженерные имена с абсолютной константностью для SwiGLU
void launch_matmul(float *output_matrix,
                   const float *matrix_a,
                   const float *matrix_b,
                   int batch_size, int out_features, int in_features);

void launch_swish_glu(float *output,
                      const float *gate_input,
                      const float *up_input,
                      int size);
}

// Запись новых векторов в статический буфер KV-Cache
void launch_update_kv_cache(float *k_cache,
                            float *v_cache,
                            const float *new_k,
                            const float *new_v,
                            int token_index, int hidden_size);

#endif
