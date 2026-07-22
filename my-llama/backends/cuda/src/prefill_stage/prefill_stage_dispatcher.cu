#include "prefill_stage.h"
#include "dequantize_fusion_mma_gguf_q4_k.cuh"
#include "data_types.h"
#include <cstdio>

extern "C" void launch_prefill_linear(
    void *output_activations,
    const void *input_activations,
    const void *quantized_weights,
    int32_t batch_size_or_tokens,
    int32_t hidden_units_out,
    int32_t hidden_units_in,
    int32_t activation_type,
    int32_t quantization_type,
    void *stream_ptr
) {

    printf("[DISPATCHER DEBUG] Intercepted launch_prefill_linear for FP4:\n");
    printf("  - Target Output Ptr: %p\n", output_activations);
    printf("  - Quantization Type: %d\n", quantization_type);
    fflush(stdout);

    const auto q_type = static_cast<QuantType>(quantization_type);

    switch (q_type) {
        case QuantType::GGUF_Q4_K:
            launch_fused_gemm_gguf_q4_k(
                output_activations,
                input_activations,
                quantized_weights,
                batch_size_or_tokens,
                hidden_units_out,
                hidden_units_in,
                activation_type,
                stream_ptr
            );
            break;
        default:
            printf("Error: Quantization type %d is not implemented or unsupported in prefill linear stage.\n", quantization_type);
            break;
    }
}
