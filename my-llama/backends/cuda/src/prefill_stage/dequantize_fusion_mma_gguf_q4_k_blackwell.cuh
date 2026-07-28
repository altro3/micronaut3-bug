#pragma once
#include <stdint.h>
#include <cuda_bf16.h>

#ifdef __cplusplus
extern "C" {
#endif

#pragma pack(push, 1)
struct BlockQ4K {
    __nv_bfloat16 d;
    __nv_bfloat16 dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};
#pragma pack(pop)

void launch_fused_gemm_gguf_blackwell_fp4_native(
    void *output_activations,
    const void *input_activations,
    const void *quantized_weights,
    int32_t batch_size_or_tokens,
    int32_t hidden_units_out,
    int32_t hidden_units_in,
    void *stream_ptr
);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
static_assert(sizeof(BlockQ4K) == 144, "GGUF structural layout validation failed! Struct size must be exactly 144 bytes.");
#endif
