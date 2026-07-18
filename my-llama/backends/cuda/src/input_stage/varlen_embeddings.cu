#include "varlen_embeddings.cuh"
#include <cuda_fp4.h>
#include <type_traits>

template<typename T>
__global__ void varlen_embeddings_fused_kernel(
    __nv_bfloat16 * __restrict__ out,
    const void * __restrict__ weight,
    const float * __restrict__ weight_scales,
    const uint32_t * __restrict__ tokens,
    const int32_t * __restrict__ seq_offsets,
    const int32_t * __restrict__ block_table,
    int32_t * __restrict__ slot_mapping,
    const int32_t max_blocks_per_seq,
    const int32_t block_size,
    const int32_t total_tokens,
    const int32_t out_features,
    const int32_t vocab_size,
    const int32_t num_seqs
) {
    const int32_t token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;

    const uint32_t token_id = tokens[token_idx];
    const int32_t tid = threadIdx.x;
    const int32_t stride = blockDim.x;

    if (tid < 32 && slot_mapping != nullptr && seq_offsets != nullptr && block_table != nullptr) {
        if (tid == 0) {
            const int32_t seq_idx = find_sequence_index(seq_offsets, num_seqs, token_idx);
            const int32_t start_tok_idx = seq_offsets[seq_idx];
            const int32_t token_local_idx = token_idx - start_tok_idx;

            const int32_t logical_block_idx = token_local_idx / block_size;
            const int32_t block_offset = token_local_idx % block_size;

            const int32_t *seq_blocks = block_table + seq_idx * max_blocks_per_seq;
            const int32_t physical_block_id = seq_blocks[logical_block_idx];

            if (physical_block_id == -1) {
                slot_mapping[token_idx] = -1;
            } else {
                slot_mapping[token_idx] = physical_block_id * block_size + block_offset;
            }
        }
    }

    __nv_bfloat16 *const out_ptr = out + static_cast<int64_t>(token_idx) * out_features;

    if (token_id >= static_cast<uint32_t>(vocab_size)) {
        const float4 zero_f4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const int32_t out_features_f4 = out_features / 8;
        for (int32_t i = tid; i < out_features_f4; i += stride) {
            __stcs(reinterpret_cast<float4 *>(&out_ptr[i * 8]), zero_f4);
        }
        const int32_t remainder_start = out_features_f4 * 8;
        for (int32_t i = remainder_start + tid; i < out_features; i += stride) {
            out_ptr[i] = __float2bfloat16(0.0f);
        }
        return;
    }

    if constexpr (std::is_same_v<T, __nv_fp4_e2m1>) {
        const uint8_t *const w_fp4 = static_cast<const uint8_t *>(weight);
        const int64_t weight_row_byte_offset = static_cast<int64_t>(token_id) * (out_features / 2);
        const int64_t scale_row_offset = static_cast<int64_t>(token_id) * (out_features / 32);
        const int32_t pairs_to_process = out_features / 2;

        for (int32_t i = tid; i < pairs_to_process; i += stride) {
            const int64_t byte_idx = weight_row_byte_offset + i;
            const uint8_t packed_val = w_fp4[byte_idx];

            const int32_t element_idx_0 = i * 2;
            const int64_t scale_idx = scale_row_offset + (element_idx_0 / 32);
            const float scale = weight_scales[scale_idx];

            __half2_raw raw_h2 = __nv_cvt_fp4x2_to_halfraw2(packed_val, __NV_E2M1);
            const float2 f2_vals = __half22float2(*reinterpret_cast<__half2 *>(&raw_h2));

            out_ptr[element_idx_0] = __float2bfloat16(f2_vals.x * scale);
            out_ptr[element_idx_0 + 1] = __float2bfloat16(f2_vals.y * scale);
        }
    } else {
        using NonFP4T = std::conditional_t<std::is_same_v<T, __nv_fp4_e2m1>, __nv_bfloat16, T>;
        const NonFP4T *const w_typed = static_cast<const NonFP4T *>(weight);
        const int64_t weight_row_offset = static_cast<int64_t>(token_id) * out_features;

        using V4 = VectorType<NonFP4T>::Type4;
        constexpr int32_t elements_per_v4 = sizeof(V4) / sizeof(NonFP4T);
        const int32_t out_features_v4 = out_features / elements_per_v4;

        for (int32_t idx_v4 = tid; idx_v4 < out_features_v4; idx_v4 += stride) {
            const int64_t offset = weight_row_offset + idx_v4 * elements_per_v4;
            const int32_t out_offset = idx_v4 * elements_per_v4;

            const V4 w0 = __ldcs(reinterpret_cast<const V4 *>(&w_typed[offset]));

#pragma unroll
            for (int32_t e = 0; e < elements_per_v4; ++e) {
                out_ptr[out_offset + e] = static_cast<__nv_bfloat16>(reinterpret_cast<const NonFP4T *>(&w0)[e]);
            }
        }

        const int32_t remainder_start = out_features_v4 * elements_per_v4;
        for (int32_t i = remainder_start + tid; i < out_features; i += stride) {
            out_ptr[i] = static_cast<__nv_bfloat16>(w_typed[weight_row_offset + i]);
        }
    }
}

template __global__ void varlen_embeddings_fused_kernel<__nv_bfloat16>(
    __nv_bfloat16 * __restrict__, const void * __restrict__, const float * __restrict__, const uint32_t * __restrict__, const int32_t * __restrict__, const int32_t * __restrict__, int32_t * __restrict__, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);

template __global__ void varlen_embeddings_fused_kernel<__nv_fp8_e4m3>(
    __nv_bfloat16 * __restrict__, const void * __restrict__, const float * __restrict__, const uint32_t * __restrict__, const int32_t * __restrict__, const int32_t * __restrict__, int32_t * __restrict__, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);

template __global__ void varlen_embeddings_fused_kernel<__nv_fp4_e2m1>(
    __nv_bfloat16 * __restrict__, const void * __restrict__, const float * __restrict__, const uint32_t * __restrict__, const int32_t * __restrict__, const int32_t * __restrict__, int32_t * __restrict__, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);
