#ifndef KERNELS_H
#define KERNELS_H

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

void launch_matmul(
    float *output_matrix,
    const float *matrix_a,
    const float *matrix_b,
    int batch_size,
    int out_features,
    int in_features,
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
}

#endif
