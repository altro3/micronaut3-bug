#include <cuda_runtime.h>
#include <stdint.h>

extern __global__ void speculative_verify_kernel(
    int32_t * __restrict__ accepted_tokens,
    int32_t * __restrict__ num_accepted,
    const float * __restrict__ target_logits,
    const float * __restrict__ draft_probs,
    const int32_t * __restrict__ draft_tokens,
    const float * __restrict__ random_nums,
    float * __restrict__ workspace,
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
    float *workspace,
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

    speculative_verify_kernel<<<blocks, threads, 0, stream>>>(
        accepted_tokens, num_accepted, target_logits, draft_probs, draft_tokens, random_nums, workspace,
        vocab_size, max_draft_tokens, temperature
    );
}

}
