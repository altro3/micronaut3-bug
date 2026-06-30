#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_cross_entropy_kernel(
    const float * __restrict__ logits,
    const int * __restrict__ targets,
    float * __restrict__ d_logits,
    float * __restrict__ losses,
    const int vocab_size
) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const float *token_logits = logits + token_idx * vocab_size;
    float *token_d_logits = d_logits + token_idx * vocab_size;
    const int target_label = targets[token_idx];

    float local_max = -INFINITY;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        if (token_logits[i] > local_max) {
            local_max = token_logits[i];
        }
    }

    extern __shared__ float s_mem[];
    s_mem[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            if (s_mem[tid + stride] > s_mem[tid]) {
                s_mem[tid] = s_mem[tid + stride];
            }
        }
        __syncthreads();
    }
    float global_max = s_mem[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        local_sum += expf(token_logits[i] - global_max);
    }

    s_mem[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_mem[tid] += s_mem[tid + stride];
        }
        __syncthreads();
    }
    const float global_sum = s_mem[0];
    __syncthreads();

    if (tid == 0) {
        const float log_p_target = token_logits[target_label] - global_max - logf(global_sum);
        losses[token_idx] = -log_p_target;
    }

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        const float p_i = expf(token_logits[i] - global_max) / global_sum;
        const float y_i = i == target_label ? 1.0f : 0.0f;

        token_d_logits[i] = p_i - y_i;
    }
}

extern "C" {
void launch_fused_cross_entropy(
    const float *logits,
    const int *targets,
    float *d_logits,
    float *losses,
    const int num_tokens,
    const int vocab_size,
    void *stream_ptr
) {
    constexpr int threads = 256;
    int blocks = num_tokens;
    size_t shared_mem_size = threads * sizeof(float);

    const auto stream = static_cast<cudaStream_t>(stream_ptr);

    fused_cross_entropy_kernel<<<blocks, threads, shared_mem_size, stream>>>(
        logits, targets, d_logits, losses, vocab_size
    );
}
}
