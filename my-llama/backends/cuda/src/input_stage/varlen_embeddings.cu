#include "varlen_embeddings.cuh"
#include <cuda_fp4.h>
#include <type_traits>
#include <stdio.h>

template<typename T>
__global__ void varlen_embeddings_fused_kernel(
    __nv_bfloat16 * __restrict__ out,
    const void * __restrict__ weight,
    const float * __restrict__ weight_scales,
    const uint32_t * __restrict__ tokens,
    const int32_t * __restrict__ seq_offsets,
    const int32_t * __restrict__ block_table,
    int32_t * __restrict__ slot_mapping,
    int32_t max_blocks_per_seq,
    int32_t block_size,
    int32_t total_tokens,
    int32_t out_features,
    int32_t vocab_size,
    int32_t num_seqs
) {
    const int32_t tid = threadIdx.x;
    const int32_t token_idx = blockIdx.x;
    const int32_t stride = blockDim.x;

    if (token_idx >= total_tokens) return;

    if (tid == 0) {
        if (slot_mapping != nullptr && seq_offsets != nullptr && block_table != nullptr) {
            const int32_t seq_idx = find_sequence_index(seq_offsets, num_seqs, token_idx);
            const int32_t start_tok_idx = seq_offsets[seq_idx];
            const int32_t token_local_idx = token_idx - start_tok_idx;
            const int32_t logical_block_idx = token_local_idx / block_size;
            const int32_t block_offset = token_local_idx % block_size;
            const int32_t physical_block_id = block_table[seq_idx * max_blocks_per_seq + logical_block_idx];

            int32_t mapped_slot = -1;
            if (physical_block_id != -1) {
                mapped_slot = physical_block_id * block_size + block_offset;
            }
            slot_mapping[token_idx] = mapped_slot;
        }
    }

    const uint32_t token_id = tokens[token_idx];
    __nv_bfloat16 *const out_ptr = out + static_cast<int64_t>(token_idx) * out_features;

    if (token_id >= static_cast<uint32_t>(vocab_size)) {
        const uint4 zero_u4 = make_uint4(0, 0, 0, 0);
        const int32_t out_features_u4 = out_features / 8;
        for (int32_t i = tid; i < out_features_u4; i += stride) {
            __stcs(reinterpret_cast<uint4 *>(&out_ptr[i * 8]), zero_u4);
        }
        const int32_t remainder_start = out_features_u4 * 8;
        for (int32_t i = remainder_start + tid; i < out_features; i += stride) {
            out_ptr[i] = __float2bfloat16(0.0f);
        }
        return;
    }

    if constexpr (std::is_same_v<T, __nv_fp4_e2m1>) {
        const uint2 *const w_u2 = static_cast<const uint2 *>(weight);
        const int64_t weight_row_u2_offset = static_cast<int64_t>(token_id) * (out_features / 16);
        const int64_t scale_row_offset = static_cast<int64_t>(token_id) * (out_features / 32);
        const int32_t u2_to_process = out_features / 16;

        for (int32_t i = tid; i < u2_to_process; i += stride) {
            const uint2 packed_val64 = __ldcs(&w_u2[weight_row_u2_offset + i]);
            const float scale = __ldcs(&weight_scales[scale_row_offset + i]);

            uint32_t p0 = packed_val64.x;
            uint32_t p1 = packed_val64.y;

            uint8_t b0 = static_cast<uint8_t>(p0 & 0xFF);
            uint8_t b1 = static_cast<uint8_t>((p0 >> 8) & 0xFF);
            uint8_t b2 = static_cast<uint8_t>((p0 >> 16) & 0xFF);
            uint8_t b3 = static_cast<uint8_t>((p0 >> 24) & 0xFF);
            uint8_t b4 = static_cast<uint8_t>(p1 & 0xFF);
            uint8_t b5 = static_cast<uint8_t>((p1 >> 8) & 0xFF);
            uint8_t b6 = static_cast<uint8_t>((p1 >> 16) & 0xFF);
            uint8_t b7 = static_cast<uint8_t>((p1 >> 24) & 0xFF);

            float2 f2_0 = __half22float2(__half2(__nv_cvt_fp4x2_to_halfraw2(b0, __NV_E2M1)));
            float2 f2_1 = __half22float2(__half2(__nv_cvt_fp4x2_to_halfraw2(b1, __NV_E2M1)));
            float2 f2_2 = __half22float2(__half2(__nv_cvt_fp4x2_to_halfraw2(b2, __NV_E2M1)));
            float2 f2_3 = __half22float2(__half2(__nv_cvt_fp4x2_to_halfraw2(b3, __NV_E2M1)));
            float2 f2_4 = __half22float2(__half2(__nv_cvt_fp4x2_to_halfraw2(b4, __NV_E2M1)));
            float2 f2_5 = __half22float2(__half2(__nv_cvt_fp4x2_to_halfraw2(b5, __NV_E2M1)));
            float2 f2_6 = __half22float2(__half2(__nv_cvt_fp4x2_to_halfraw2(b6, __NV_E2M1)));
            float2 f2_7 = __half22float2(__half2(__nv_cvt_fp4x2_to_halfraw2(b7, __NV_E2M1)));

            uint4 out_v0, out_v1;
            *reinterpret_cast<__nv_bfloat162 *>(&out_v0.x) = __floats2bfloat162_rn(f2_0.x * scale, f2_0.y * scale);
            *reinterpret_cast<__nv_bfloat162 *>(&out_v0.y) = __floats2bfloat162_rn(f2_1.x * scale, f2_1.y * scale);
            *reinterpret_cast<__nv_bfloat162 *>(&out_v0.z) = __floats2bfloat162_rn(f2_2.x * scale, f2_2.y * scale);
            *reinterpret_cast<__nv_bfloat162 *>(&out_v0.w) = __floats2bfloat162_rn(f2_3.x * scale, f2_3.y * scale);
            *reinterpret_cast<__nv_bfloat162 *>(&out_v1.x) = __floats2bfloat162_rn(f2_4.x * scale, f2_4.y * scale);
            *reinterpret_cast<__nv_bfloat162 *>(&out_v1.y) = __floats2bfloat162_rn(f2_5.x * scale, f2_5.y * scale);
            *reinterpret_cast<__nv_bfloat162 *>(&out_v1.z) = __floats2bfloat162_rn(f2_6.x * scale, f2_6.y * scale);
            *reinterpret_cast<__nv_bfloat162 *>(&out_v1.w) = __floats2bfloat162_rn(f2_7.x * scale, f2_7.y * scale);

            const int32_t out_base = i * 16;
            __stcs(reinterpret_cast<uint4 *>(&out_ptr[out_base]), out_v0);
            __stcs(reinterpret_cast<uint4 *>(&out_ptr[out_base + 8]), out_v1);
        }
    } else {
        const auto w_u4 = static_cast<const uint4 *>(weight);

        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            const int32_t out_features_v4 = out_features >> 3;
            const int64_t weight_row_u4_offset = static_cast<int64_t>(token_id) * out_features_v4;

            for (int32_t idx_v4 = tid; idx_v4 < out_features_v4; idx_v4 += stride) {
                const uint4 w0 = __ldcs(&w_u4[weight_row_u4_offset + idx_v4]);
                const int32_t out_offset = idx_v4 << 3;
                __stcs(reinterpret_cast<uint4 *>(&out_ptr[out_offset]), w0);
            }
        } else if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
            const int32_t out_features_v4 = out_features >> 4;
            const int64_t weight_row_u4_offset = static_cast<int64_t>(token_id) * out_features_v4;

            for (int32_t idx_v4 = tid; idx_v4 < out_features_v4; idx_v4 += stride) {
                const uint4 w0 = __ldcs(&w_u4[weight_row_u4_offset + idx_v4]);
                const int32_t out_offset = idx_v4 << 4;

                const uint32_t *const packed_vals = reinterpret_cast<const uint32_t *>(&w0);

                float2 f2_0 = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(static_cast<uint16_t>(packed_vals[0] & 0xFFFF), __NV_E4M3)));
                float2 f2_1 = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(static_cast<uint16_t>((packed_vals[0] >> 16) & 0xFFFF), __NV_E4M3)));
                float2 f2_2 = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(static_cast<uint16_t>(packed_vals[1] & 0xFFFF), __NV_E4M3)));
                float2 f2_3 = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(static_cast<uint16_t>((packed_vals[1] >> 16) & 0xFFFF), __NV_E4M3)));
                float2 f2_4 = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(static_cast<uint16_t>(packed_vals[2] & 0xFFFF), __NV_E4M3)));
                float2 f2_5 = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(static_cast<uint16_t>((packed_vals[2] >> 16) & 0xFFFF), __NV_E4M3)));
                float2 f2_6 = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(static_cast<uint16_t>(packed_vals[3] & 0xFFFF), __NV_E4M3)));
                float2 f2_7 = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(static_cast<uint16_t>((packed_vals[3] >> 16) & 0xFFFF), __NV_E4M3)));

                uint4 final_u4_0;
                *reinterpret_cast<__nv_bfloat162 *>(&final_u4_0.x) = __floats2bfloat162_rn(f2_0.x, f2_0.y);
                *reinterpret_cast<__nv_bfloat162 *>(&final_u4_0.y) = __floats2bfloat162_rn(f2_1.x, f2_1.y);
                *reinterpret_cast<__nv_bfloat162 *>(&final_u4_0.z) = __floats2bfloat162_rn(f2_2.x, f2_2.y);
                *reinterpret_cast<__nv_bfloat162 *>(&final_u4_0.w) = __floats2bfloat162_rn(f2_3.x, f2_3.y);

                uint4 final_u4_1;
                *reinterpret_cast<__nv_bfloat162 *>(&final_u4_1.x) = __floats2bfloat162_rn(f2_4.x, f2_4.y);
                *reinterpret_cast<__nv_bfloat162 *>(&final_u4_1.y) = __floats2bfloat162_rn(f2_5.x, f2_5.y);
                *reinterpret_cast<__nv_bfloat162 *>(&final_u4_1.z) = __floats2bfloat162_rn(f2_6.x, f2_6.y);
                *reinterpret_cast<__nv_bfloat162 *>(&final_u4_1.w) = __floats2bfloat162_rn(f2_7.x, f2_7.y);

                __stcs(reinterpret_cast<uint4 *>(&out_ptr[out_offset]), final_u4_0);
                __stcs(reinterpret_cast<uint4 *>(&out_ptr[out_offset + 8]), final_u4_1);
            }
        }
    }
}

void run_varlen_embeddings_bf16(
    __nv_bfloat16 *out, const void *weight, const float *weight_scales, const uint32_t *tokens,
    const int32_t *seq_offsets, const int32_t *block_table, int32_t *slot_mapping,
    int32_t max_blocks, int32_t b_size, int32_t t_tokens, int32_t out_f, int32_t v_size,
    int32_t n_seqs, int32_t tpb, cudaStream_t stream
) {
    varlen_embeddings_fused_kernel<__nv_bfloat16><<<t_tokens, tpb, 0, stream>>>(
        out, weight, weight_scales, tokens, seq_offsets, block_table, slot_mapping,
        max_blocks, b_size, t_tokens, out_f, v_size, n_seqs
    );
}

void run_varlen_embeddings_fp8(
    __nv_bfloat16 *out, const void *weight, const float *weight_scales, const uint32_t *tokens,
    const int32_t *seq_offsets, const int32_t *block_table, int32_t *slot_mapping,
    int32_t max_blocks, int32_t b_size, int32_t t_tokens, int32_t out_f, int32_t v_size,
    int32_t n_seqs, int32_t tpb, cudaStream_t stream
) {
    varlen_embeddings_fused_kernel<__nv_fp8_e4m3><<<t_tokens, tpb, 0, stream>>>(
        out, weight, weight_scales, tokens, seq_offsets, block_table, slot_mapping,
        max_blocks, b_size, t_tokens, out_f, v_size, n_seqs
    );
}

void run_varlen_embeddings_fp4(
    __nv_bfloat16 *out, const void *weight, const float *weight_scales, const uint32_t *tokens,
    const int32_t *seq_offsets, const int32_t *block_table, int32_t *slot_mapping,
    int32_t max_blocks, int32_t b_size, int32_t t_tokens, int32_t out_f, int32_t v_size,
    int32_t n_seqs, int32_t tpb, cudaStream_t stream
) {
    varlen_embeddings_fused_kernel<__nv_fp4_e2m1><<<t_tokens, tpb, 0, stream>>>(
        out, weight, weight_scales, tokens, seq_offsets, block_table, slot_mapping,
        max_blocks, b_size, t_tokens, out_f, v_size, n_seqs
    );
}
