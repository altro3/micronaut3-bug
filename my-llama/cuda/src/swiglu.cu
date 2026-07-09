#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void swish_glu_fused_vectorized_kernel(
    float4 * __restrict__ output,
    const float4 * __restrict__ gate_input,
    const float4 * __restrict__ up_input,
    const int size_v4
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size_v4) {
        const float4 g_v4 = __ldcs(&gate_input[idx]);
        const float4 u_v4 = __ldcs(&up_input[idx]);
        float4 out_v4;

        const float swish_x = g_v4.x * __frcp_rn(1.0f + __expf(-g_v4.x));
        out_v4.x = swish_x * u_v4.x;

        const float swish_y = g_v4.y * __frcp_rn(1.0f + __expf(-g_v4.y));
        out_v4.y = swish_y * u_v4.y;

        const float swish_z = g_v4.z * __frcp_rn(1.0f + __expf(-g_v4.z));
        out_v4.z = swish_z * u_v4.z;

        const float swish_w = g_v4.w * __frcp_rn(1.0f + __expf(-g_v4.w));
        out_v4.w = swish_w * u_v4.w;

        __stcs(&output[idx], out_v4);
    }
}

__global__ void swish_glu_fused_scalar_kernel(
    float * __restrict__ output,
    const float * __restrict__ gate_input,
    const float * __restrict__ up_input,
    const int size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        const float g = __ldcs(&gate_input[idx]);
        const float u = __ldcs(&up_input[idx]);
        const float swish = g * __frcp_rn(1.0f + __expf(-g));
        __stcs(&output[idx], swish * u);
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
    const auto stream = static_cast<cudaStream_t>(stream_ptr);
    constexpr int threads = 256;

    const bool is_aligned = reinterpret_cast<uintptr_t>(output) % 16 == 0 &&
                            reinterpret_cast<uintptr_t>(gate_input) % 16 == 0 &&
                            reinterpret_cast<uintptr_t>(up_input) % 16 == 0;

    if (size % 4 == 0 && is_aligned) {
        const int size_v4 = size / 4;
        const int blocks = (size_v4 + threads - 1) / threads;
        swish_glu_fused_vectorized_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<float4 *>(output),
            reinterpret_cast<const float4 *>(gate_input),
            reinterpret_cast<const float4 *>(up_input),
            size_v4
        );
    } else {
        const int blocks = (size + threads - 1) / threads;
        swish_glu_fused_scalar_kernel<<<blocks, threads, 0, stream>>>(
            output, gate_input, up_input, size
        );
    }
}
}
