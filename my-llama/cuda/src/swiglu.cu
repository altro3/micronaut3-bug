#include <cuda_runtime.h>

__global__ void swish_glu_fused_kernel(
    float * __restrict__ output,
    const float * __restrict__ gate_input,
    const float * __restrict__ up_input,
    const int size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        const float g = gate_input[idx];
        const float u = up_input[idx];
        const float swish = g / (1.0f + expf(-g));
        output[idx] = swish * u;
    }
}

extern "C" {
void launch_swish_glu(
    float *output,
    const float *gate_input,
    const float *up_input,
    const int size,
    void *stream_ptr
) {
    constexpr int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    const auto stream = static_cast<cudaStream_t>(stream_ptr);

    swish_glu_fused_kernel<<<blocks, threads, 0, stream>>>(output, gate_input, up_input, size);
}
}
