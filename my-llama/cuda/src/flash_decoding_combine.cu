#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

__global__ void paged_flash_decoding_combine_kernel(
    float * __restrict__ output,
    const float * __restrict__ partial_out,
    const float * __restrict__ partial_max,
    const float * __restrict__ partial_sum,
    const int num_heads,
    const int head_dim,
    const int num_chunks
) {
    const int head_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int tid = threadIdx.x;

    const int head_dim_f4 = head_dim / 4;
    if (tid >= head_dim_f4) return;

    float global_max = -1e20f;
    float global_sum = 0.0f;

    const long long batch_head_offset = ((static_cast<long long>(seq_idx) * num_heads) + head_idx) * num_chunks;

    for (int c = 0; c < num_chunks; ++c) {
        const long long partial_offset = batch_head_offset + c;
        const float p_max = partial_max[partial_offset];
        const float p_sum = partial_sum[partial_offset];

        const float old_max = global_max;
        if (p_max > global_max) {
            global_max = p_max;
            global_sum = global_sum * __expf(old_max - p_max) + p_sum;
        } else {
            global_sum += p_sum * __expf(p_max - global_max);
        }
    }

    const float inv_global_sum = 1.0f / (global_sum + 1e-9f);
    float4 out_acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (int c = 0; c < num_chunks; ++c) {
        const long long partial_offset = batch_head_offset + c;
        const float p_max = partial_max[partial_offset];
        const float p_sum = partial_sum[partial_offset];

        const float rescale_factor = p_sum * __expf(p_max - global_max) * inv_global_sum;
        const float *const p_out_ptr = partial_out + partial_offset * head_dim;
        const float4 p_val = *reinterpret_cast<const float4 *>(&p_out_ptr[tid * 4]);

        out_acc.x += p_val.x * rescale_factor;
        out_acc.y += p_val.y * rescale_factor;
        out_acc.z += p_val.z * rescale_factor;
        out_acc.w += p_val.w * rescale_factor;
    }

    const long long out_offset = ((static_cast<long long>(seq_idx) * num_heads) + head_idx) * head_dim;
    float *const final_out_ptr = output + out_offset;
    *reinterpret_cast<float4 *>(&final_out_ptr[tid * 4]) = out_acc;
}
