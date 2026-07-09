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
    void *stream
);
}

#endif
