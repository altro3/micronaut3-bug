#ifndef KERNELS_H
#define KERNELS_H

extern "C" {
void test_cuda_setup(float *d_array, int size);

void launch_rms_norm(float *output,
                     const float *input,
                     const float *weight,
                     int batch_size,
                     int hidden_size,
                     float epsilon);
}

#endif
