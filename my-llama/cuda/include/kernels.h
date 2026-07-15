#ifndef KERNELS_H
#define KERNELS_H

struct MatmulContext;

extern "C" {
void launch_adamw(
    float *weights,
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
    void *stream_ptr
);

void launch_swish_glu(
    float *output,
    const float *gate_input,
    const float *up_input,
    int size,
    void *stream_ptr
);

void launch_rms_norm(
    float *output,
    const float *input,
    const float *weight,
    int batch_size,
    int hidden_size,
    float epsilon,
    void *stream_ptr
);

MatmulContext *create_matmul_context(size_t workspace_size);

void destroy_matmul_context(MatmulContext *ctx);

void launch_matmul_universal(
    MatmulContext *ctx,
    void *output_matrix,
    const void *matrix_a,
    const void *matrix_b,
    int batch_size,
    int out_features,
    int in_features,
    int dtype_int,
    int epilogue_int,
    const float *a_scale_ptr,
    const float *b_scale_ptr,
    void *stream_ptr
);

void launch_matmul_gguf_q4_k(
    float *output,
    const void *weights,
    const float *vec_x,
    int out_features,
    int in_features,
    void *stream_ptr
);

void launch_embeddings(
    float *out,
    const float *weight,
    const unsigned int *tokens,
    int total_tokens,
    int out_features,
    int vocab_size,
    void *stream_ptr
);

void launch_fused_attention(
    float *output,
    const float *query,
    const float *k_cache,
    const float *v_cache,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int current_seq_len,
    void *stream_ptr
);

void launch_rope_forward(
    float *vec,
    const int *positions,
    int num_heads,
    int head_dim,
    int total_tokens,
    void *stream_ptr
);

void launch_rope_backward(
    float *grad_in,
    const int *positions,
    int num_heads,
    int head_dim,
    int total_tokens,
    void *stream_ptr
);

void launch_cross_entropy_loss(
    const float *logits,
    float *grads,
    const int *targets,
    float *losses,
    int total_tokens,
    int vocab_size,
    void *stream_ptr
);

void launch_flash_decoding(
    float *output,
    float *partial_out,
    float *partial_max,
    float *partial_sum, 
    const float *query,
    const uint8_t *k_cache,
    const uint8_t *v_cache,
    const float *k_scales,
    const float *v_scales,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int current_seq_len,
    int chunk_size,
    void *stream_ptr
);
}

#endif
