#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" {
__global__ void swish_glu_scalar_fallback_kernel(
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

__global__ void __launch_bounds__(256, 4) swish_glu_ultimate_blackwell_kernel(
    float4 * __restrict__ output,
    const float4 * __restrict__ gate_input,
    const float4 * __restrict__ up_input,
    const int size_v4,
    const int total_elements
) {
    const int stride = blockDim.x * gridDim.x;
    int idx_v4 = blockIdx.x * blockDim.x + threadIdx.x;

#pragma unroll
    for (int step = 0; step < 2; ++step) {
        if (idx_v4 < size_v4) {
            const float4 g = __ldcs(&gate_input[idx_v4]);
            const float4 u = __ldcs(&up_input[idx_v4]);
            float4 out_v4;

            out_v4.x = __expf(-g.x);
            out_v4.y = __expf(-g.y);
            out_v4.z = __expf(-g.z);
            out_v4.w = __expf(-g.w);

            out_v4.x = __frcp_rn(1.0f + out_v4.x);
            out_v4.y = __frcp_rn(1.0f + out_v4.y);
            out_v4.z = __frcp_rn(1.0f + out_v4.z);
            out_v4.w = __frcp_rn(1.0f + out_v4.w);

            out_v4.x = g.x * out_v4.x * u.x;
            out_v4.y = g.y * out_v4.y * u.y;
            out_v4.z = g.z * out_v4.z * u.z;
            out_v4.w = g.w * out_v4.w * u.w;

            __stcs(&output[idx_v4], out_v4);
        }
        idx_v4 += stride;
    }

    if (blockIdx.x == gridDim.x - 1) {
        const int scalar_idx = size_v4 * 4 + threadIdx.x;
        if (scalar_idx < total_elements) {
            float *s_output = reinterpret_cast<float *>(output);
            const float *s_gate = reinterpret_cast<const float *>(gate_input);
            const float *s_up = reinterpret_cast<const float *>(up_input);

            const float g = __ldcs(&s_gate[scalar_idx]);
            const float u = __ldcs(&s_up[scalar_idx]);
            const float swish = g * __frcp_rn(1.0f + __expf(-g));
            __stcs(&s_output[scalar_idx], swish * u);
        }
    }
}

extern "C" void launch_swish_glu(
    float *output,
    const float *gate_input,
    const float *up_input,
    const int size,
    void *stream_ptr
) {
    const auto stream = static_cast<cudaStream_t>(stream_ptr);
    constexpr int threads = 256;

    const bool is_aligned = reinterpret_cast<uintptr_t>(output) % 16 == 0
                            && reinterpret_cast<uintptr_t>(gate_input) % 16 == 0
                            && reinterpret_cast<uintptr_t>(up_input) % 16 == 0;

    if (is_aligned) [[likely]] {
        const int size_v4 = size / 4;
        const int total_threads_needed = (size_v4 + 1) / 2;
        int blocks = (total_threads_needed + threads - 1) / threads;
        if (blocks < 1) blocks = 1;

        swish_glu_ultimate_blackwell_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<float4 *>(output),
            reinterpret_cast<const float4 *>(gate_input),
            reinterpret_cast<const float4 *>(up_input),
            size_v4, size
        );
    } else [[unlikely]] {
        const int blocks = (size + threads - 1) / threads;
        swish_glu_scalar_fallback_kernel<<<blocks, threads, 0, stream>>>(
            output, gate_input, up_input, size
        );
    }
}
}
