#ifndef KERNELS_H
#define KERNELS_H

extern "C" {
// =========================================================================
// Системные и инфраструктурные функции фреймворка
// =========================================================================
void test_cuda_setup(float *d_array, int size);

void init_cublas_infrastructure();

// =========================================================================
// Прямой проход модели (Forward Pass) — Строго асинхронные стримы
// =========================================================================
void launch_rms_norm(float *output,
                     const float *input,
                     const float *weight,
                     int batch_size,
                     int hidden_size,
                     float epsilon,
                     void *stream);

void launch_matmul(float *output_matrix,
                   const float *matrix_a,
                   const float *matrix_b,
                   int batch_size, int out_features, int in_features,
                   void *stream);

void launch_embeddings(float *out, const float *weight, const unsigned int *tokens, int total_tokens, int out_features, int vocab_size, cudaStream_t stream);

void launch_swish_glu(float *output,
                      const float *gate_input,
                      const float *up_input,
                      int size,
                      void *stream);

// =========================================================================
// Подсистема Многоголового Внимания (Multi-Query / Grouped-Query Attention)
// =========================================================================
void launch_update_kv_cache(float *k_cache,
                            float *v_cache,
                            const float *new_k,
                            const float *new_v,
                            int token_index, int hidden_size,
                            void *stream_ptr);

void launch_attention_scores(float *output_scores,
                             const float *query,
                             const float *k_cache,
                             int num_heads, int num_kv_heads, int head_dim, int current_seq_len,
                             void *stream);

void launch_softmax_attention(float *scores,
                              int num_heads, int current_seq_len,
                              void *stream);

void launch_attention_values(float *output,
                             const float *probabilities,
                             const float *v_cache,
                             int num_heads, int num_kv_heads, int head_dim, int current_seq_len,
                             void *stream);

// =========================================================================
// Инференс и генерация текста (Inference)
// =========================================================================
void launch_argmax(int *output_index, const float *logits, int vocab_size, void *stream);

// =========================================================================
// Обучение и Обратный проход (Training & Backward Pass)
// =========================================================================
void launch_fused_cross_entropy(
    const float *logits,
    const int *targets,
    float *d_logits,
    float *losses,
    int num_tokens,
    int vocab_size,
    void *stream_ptr
);

void launch_matmul_backward_weights(
    float *d_weights,
    const float *input,
    const float *d_output,
    int batch_size,
    int out_features,
    int in_features,
    void *stream_ptr
);

void launch_matmul_backward_input(
    float *d_input,
    const float *d_output,
    const float *weights,
    int batch_size,
    int out_features,
    int in_features,
    void *stream
);

void launch_adamw(float *weights,
                  float *gradients,
                  float *m_buffer,
                  float *v_buffer,
                  int size,
                  float lr,
                  float beta1,
                  float beta2,
                  float epsilon,
                  float weight_decay,
                  float step,
                  void *stream);
}

#endif
