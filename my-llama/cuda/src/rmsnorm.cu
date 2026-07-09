#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void rms_norm_vectorized_kernel(
    float4 * __restrict__ output,
    const float4 * __restrict__ input,
    const float4 * __restrict__ weight,
    const int hidden_size_v4,
    const float epsilon,
    const float inv_hidden_size
) {
    const int row_idx = blockIdx.x;
    const float4 *const x_v4 = input + row_idx * hidden_size_v4;
    float4 *const y_v4 = output + row_idx * hidden_size_v4;

    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int num_warps = blockDim.x / 32;

    __shared__ float s_warp_sums[32];

    float thread_sum = 0.0f;
    for (int i = tid; i < hidden_size_v4; i += blockDim.x) {
        const float4 in_val = __ldcs(&x_v4[i]);
        thread_sum += in_val.x * in_val.x;
        thread_sum += in_val.y * in_val.y;
        thread_sum += in_val.z * in_val.z;
        thread_sum += in_val.w * in_val.w;
    }

    const float warp_sum = warp_reduce_sum(thread_sum);
    if (lane_id == 0) {
        s_warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_sum = 0.0f;
        const float val = tid < num_warps ? s_warp_sums[lane_id] : 0.0f;
        block_sum = warp_reduce_sum(val);
        if (tid == 0) {
            s_warp_sums[0] = __frsqrt_rn(block_sum * inv_hidden_size + epsilon);
        }
    }
    __syncthreads();

    const float rms_inv = s_warp_sums[0];

    for (int i = tid; i < hidden_size_v4; i += blockDim.x) {
        const float4 in_val = __ldcs(&x_v4[i]);
        const float4 w_val = __ldcs(&weight[i]);
        float4 out_val;

        out_val.x = in_val.x * rms_inv * w_val.x;
        out_val.y = in_val.y * rms_inv * w_val.y;
        out_val.z = in_val.z * rms_inv * w_val.z;
        out_val.w = in_val.w * rms_inv * w_val.w;

        __stcs(&y_v4[i], out_val);
    }
}

// Тот самый резервный контур, который мы добавили в файл rmsnorm.cu:
__global__ void rms_norm_scalar_kernel(
    float * __restrict__ output,
    const float * __restrict__ input,
    const float * __restrict__ weight,
    const int hidden_size,
    const float epsilon,
    const float inv_hidden_size
) {
    const int row_idx = blockIdx.x;
    const float *const x = input + row_idx * hidden_size;
    float *const y = output + row_idx * hidden_size;

    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int num_warps = blockDim.x / 32;

    __shared__ float s_warp_sums[8];

    float thread_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        const float val = __ldcs(&x[i]);
        thread_sum += val * val;
    }

    const float warp_sum = warp_reduce_sum(thread_sum);

    if (lane_id == 0) {
        s_warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        const float val = (tid < num_warps) ? s_warp_sums[lane_id] : 0.0f;
        const float block_sum = warp_reduce_sum(val);
        if (tid == 0) {
            s_warp_sums[0] = __frsqrt_rn(block_sum * inv_hidden_size + epsilon);
        }
    }
    __syncthreads();

    const float rms_inv = s_warp_sums[0];

    for (int i = tid; i < hidden_size; i += blockDim.x) {
        __stcs(&y[i], __ldcs(&x[i]) * rms_inv * __ldcs(&weight[i]));
    }
}

extern "C" {
void launch_rms_norm(
    float *output,
    const float *input,
    const float *weight,
    const int batch_size,
    const int hidden_size,
    const float epsilon,
    void *stream_ptr
) {
    const auto stream = static_cast<cudaStream_t>(stream_ptr);
    const float inv_hidden_size = 1.0f / static_cast<float>(hidden_size);

    const bool is_aligned = reinterpret_cast<uintptr_t>(output) % 16 == 0 &&
                            reinterpret_cast<uintptr_t>(input) % 16 == 0 &&
                            reinterpret_cast<uintptr_t>(weight) % 16 == 0;

    constexpr int threads = 256;
    const int blocks = batch_size;

    if (hidden_size % 4 == 0 && is_aligned) {
        const int hidden_size_v4 = hidden_size / 4;

        rms_norm_vectorized_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<float4 *>(output),
            reinterpret_cast<const float4 *>(input),
            reinterpret_cast<const float4 *>(weight),
            hidden_size_v4, epsilon, inv_hidden_size
        );
    } else {
        rms_norm_scalar_kernel<<<blocks, threads, 0, stream>>>(
            output, input, weight, hidden_size, epsilon, inv_hidden_size
        );
    }
}
}
