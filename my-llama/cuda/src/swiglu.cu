#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" {
__global__ void swish_glu_scalar_fallback_kernel(
    float * __restrict__ output,
    const float * __restrict__ gate_input,
    const float * __restrict__ up_input,
    const int start_idx,
    const int size
) {
    const int idx = start_idx + (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < size) {
        const float g = __ldcs(&gate_input[idx]);
        const float u = __ldcs(&up_input[idx]);
        const float swish = g / (1.0f + __expf(-g));
        __stcs(&output[idx], swish * u);
    }
}

__global__ void __launch_bounds__(256, 4) swish_glu_ultimate_blackwell_kernel(
    float4 * __restrict__ output,
    const float4 * __restrict__ gate_input,
    const float4 * __restrict__ up_input,
    const int size_v4
) {
    const int stride = blockDim.x * gridDim.x;

    for (int idx_v4 = blockIdx.x * blockDim.x + threadIdx.x; idx_v4 < size_v4; idx_v4 += stride) {
        const float4 g = __ldcs(&gate_input[idx_v4]);
        const float4 u = __ldcs(&up_input[idx_v4]);

        const float ex = __expf(-g.x);
        const float ey = __expf(-g.y);
        const float ez = __expf(-g.z);
        const float ew = __expf(-g.w);

        float4 out_v4;
        out_v4.x = g.x / (1.0f + ex) * u.x;
        out_v4.y = g.y / (1.0f + ey) * u.y;
        out_v4.z = g.z / (1.0f + ez) * u.z;
        out_v4.w = g.w / (1.0f + ew) * u.w;

        __stcs(&output[idx_v4], out_v4);
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
        int blocks = (size_v4 + threads - 1) / threads;

        if (blocks > 2048) blocks = 2048;
        if (blocks < 1) blocks = 1;

        swish_glu_ultimate_blackwell_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<float4 *>(output),
            reinterpret_cast<const float4 *>(gate_input),
            reinterpret_cast<const float4 *>(up_input),
            size_v4
        );

        const int remaining_elements = size % 4;
        if (remaining_elements > 0) {
            const int scalar_start = size_v4 * 4;
            swish_glu_scalar_fallback_kernel<<<1, threads, 0, stream>>>(
                output, gate_input, up_input,
                scalar_start, size
            );
        }
    } else [[unlikely]] {
        const int blocks = (size + threads - 1) / threads;
        swish_glu_scalar_fallback_kernel<<<blocks, threads, 0, stream>>>(
            output, gate_input, up_input,
            0, size
        );
    }
}
}
