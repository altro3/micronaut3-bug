#include <cuda_runtime.h>
#include <stdint.h>

extern __global__ void speculative_verify_kernel(
    int32_t * __restrict__ accepted_tokens,
    int32_t * __restrict__ num_accepted,
    const float * __restrict__ target_logits,
    const float * __restrict__ draft_probs,
    const int32_t * __restrict__ draft_tokens,
    const float * __restrict__ random_nums,
    int vocab_size,
    int max_draft_tokens,
    float temperature
);

extern "C" {
void launch_speculative_verify(
    int32_t *accepted_tokens,
    int32_t *num_accepted,
    const float *target_logits,
    const float *draft_probs,
    const int32_t *draft_tokens,
    const float *random_nums,
    const int num_seqs,
    const int vocab_size,
    const int max_draft_tokens,
    const float temperature,
    void *stream_ptr
) {
    if (num_seqs == 0) return;

    const auto stream = static_cast<cudaStream_t>(stream_ptr);

    constexpr int threads = 256;
    dim3 blocks(1, num_seqs);
    const size_t shared_mem_size = vocab_size * sizeof(float);

    speculative_verify_kernel<<<blocks, threads, shared_mem_size, stream>>>(
        accepted_tokens, num_accepted, target_logits, draft_probs, draft_tokens, random_nums,
        vocab_size, max_draft_tokens, temperature
    );
}
}
