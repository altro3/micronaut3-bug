#pragma once
#include <stdint.h>
#include "core_api.h"

extern "C" KERNEL_API void launch_prefill_linear(
    void *output_activations,
    const void *input_activations,
    const void *quantized_weights,
    int32_t batch_size_or_tokens,
    int32_t hidden_units_out,
    int32_t hidden_units_in,
    int32_t activation_type,
    int32_t quantization_type,
    void *stream_ptr
);
