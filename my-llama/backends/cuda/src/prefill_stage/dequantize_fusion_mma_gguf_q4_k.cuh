#pragma once
#include <stdint.h>
#include <cuda_bf16.h>

struct __align__(16) BlockQ4K {
    __nv_bfloat16 d;
    __nv_bfloat16 dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

#ifdef __cplusplus
extern "C" {
#endif

void launch_fused_gemm_gguf_q4_k(
    void *output_activations,
    const void *input_activations,
    const void *quantized_weights,
    int32_t batch_size_or_tokens,
    int32_t hidden_units_out,
    int32_t hidden_units_in,
    int32_t data_type,
    void *stream_ptr
);

#ifdef __cplusplus
}
#endif
