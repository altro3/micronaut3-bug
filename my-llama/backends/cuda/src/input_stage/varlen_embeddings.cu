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
    int32_t num_seqs,
    bool dump_debug
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

            if (dump_debug && (token_idx == 0 || token_idx == 16)) {
                printf("[CUDA DBG] token_idx: %d | seq_idx: %d | start_tok: %d | local_idx: %d | phys_blk: %d | slot: %d\n",
                       token_idx, seq_idx, start_tok_idx, token_local_idx, physical_block_id, mapped_slot);
            }
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
        // Читаем память через uint32_t вместо uint4, чтобы Blackwell не ругался на выравнивание 128-битных векторов
        const uint32_t *const w_u32 = static_cast<const uint32_t *>(weight);
        const int64_t weight_row_u32_offset = static_cast<int64_t>(token_id) * (out_features / 8);
        const int64_t scale_row_offset = static_cast<int64_t>(token_id) * (out_features / 32);

        // Каждый шаг итерации теперь обрабатывает один uint32_t (8 элементов FP4)
        const int32_t u32_to_process = out_features / 8;

        for (int32_t i = tid; i < u32_to_process; i += stride) {
            const uint32_t packed_val32 = w_u32[weight_row_u32_offset + i];

            // Вычисляем, к какой группе из 32 элементов относится этот uint32_t, чтобы взять правильную шкалу
            const int32_t group_idx = i / 4;
            const float scale = weight_scales[scale_row_offset + group_idx];

            uint8_t byte0 = static_cast<uint8_t>(packed_val32 & 0xFF);
            uint8_t byte1 = static_cast<uint8_t>((packed_val32 >> 8) & 0xFF);
            uint8_t byte2 = static_cast<uint8_t>((packed_val32 >> 16) & 0xFF);
            uint8_t byte3 = static_cast<uint8_t>((packed_val32 >> 24) & 0xFF);

            __half2_raw raw_h2_0 = __nv_cvt_fp4x2_to_halfraw2(byte0, __NV_E2M1);
            __half2_raw raw_h2_1 = __nv_cvt_fp4x2_to_halfraw2(byte1, __NV_E2M1);
            __half2_raw raw_h2_2 = __nv_cvt_fp4x2_to_halfraw2(byte2, __NV_E2M1);
            __half2_raw raw_h2_3 = __nv_cvt_fp4x2_to_halfraw2(byte3, __NV_E2M1);

            float2 f2_0 = __half22float2(*reinterpret_cast<__half2 *>(&raw_h2_0));
            float2 f2_1 = __half22float2(*reinterpret_cast<__half2 *>(&raw_h2_1));
            float2 f2_2 = __half22float2(*reinterpret_cast<__half2 *>(&raw_h2_2));
            float2 f2_3 = __half22float2(*reinterpret_cast<__half2 *>(&raw_h2_3));

            __nv_bfloat162 res0 = __floats2bfloat162_rn(f2_0.x * scale, f2_0.y * scale);
            __nv_bfloat162 res1 = __floats2bfloat162_rn(f2_1.x * scale, f2_1.y * scale);
            __nv_bfloat162 res2 = __floats2bfloat162_rn(f2_2.x * scale, f2_2.y * scale);
            __nv_bfloat162 res3 = __floats2bfloat162_rn(f2_3.x * scale, f2_3.y * scale);

            // Пишем в выходной буфер out_ptr (выходной тип по-прежнему BF16, пишем парами через uint2 = 8 байт)
            const int32_t out_base = i * 8;
            uint2 final_u2_0;
            *reinterpret_cast<__nv_bfloat162 *>(&final_u2_0.x) = res0;
            *reinterpret_cast<__nv_bfloat162 *>(&final_u2_0.y) = res1;

            uint2 final_u2_1;
            *reinterpret_cast<__nv_bfloat162 *>(&final_u2_1.x) = res2;
            *reinterpret_cast<__nv_bfloat162 *>(&final_u2_1.y) = res3;

            __stcs(reinterpret_cast<uint2 *>(&out_ptr[out_base]), final_u2_0);
            __stcs(reinterpret_cast<uint2 *>(&out_ptr[out_base + 4]), final_u2_1);
        }
    } else {
        const auto w_u4 = static_cast<const uint4 *>(weight);

        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            const int32_t out_features_v4 = out_features / 8;
            const int64_t weight_row_u4_offset = static_cast<int64_t>(token_id) * out_features_v4;

            for (int32_t idx_v4 = tid; idx_v4 < out_features_v4; idx_v4 += stride) {
                const uint4 w0 = __ldcs(&w_u4[weight_row_u4_offset + idx_v4]);
                const int32_t out_offset = idx_v4 * 8;

                auto out_bf162 = reinterpret_cast<__nv_bfloat162 *>(&out_ptr[out_offset]);

                out_bf162[0] = *reinterpret_cast<const __nv_bfloat162 *>(&w0.x);
                out_bf162[1] = *reinterpret_cast<const __nv_bfloat162 *>(&w0.y);
                out_bf162[2] = *reinterpret_cast<const __nv_bfloat162 *>(&w0.z);
                out_bf162[3] = *reinterpret_cast<const __nv_bfloat162 *>(&w0.w);
            }
        } else if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
            const int32_t out_features_v4 = out_features / 16;
            const int64_t weight_row_u4_offset = static_cast<int64_t>(token_id) * out_features_v4;

            for (int32_t idx_v4 = tid; idx_v4 < out_features_v4; idx_v4 += stride) {
                const uint4 w0 = __ldcs(&w_u4[weight_row_u4_offset + idx_v4]);
                const int32_t out_offset = idx_v4 * 16;

                auto packed_vals = reinterpret_cast<const uint32_t *>(&w0);

                for (int32_t w = 0; w < 4; ++w) {
                    uint32_t val32 = packed_vals[w];
                    uint16_t low16 = static_cast<uint16_t>(val32 & 0xFFFF);
                    uint16_t high16 = static_cast<uint16_t>((val32 >> 16) & 0xFFFF);

                    float2 f2_low = __half22float2(__nv_cvt_fp8x2_to_halfraw2(low16, __NV_E4M3));
                    float2 f2_high = __half22float2(__nv_cvt_fp8x2_to_halfraw2(high16, __NV_E4M3));

                    __nv_bfloat162 res_low = __floats2bfloat162_rn(f2_low.x, f2_low.y);
                    __nv_bfloat162 res_high = __floats2bfloat162_rn(f2_high.x, f2_high.y);

                    uint2 final_u2;
                    *reinterpret_cast<__nv_bfloat162 *>(&final_u2.x) = res_low;
                    *reinterpret_cast<__nv_bfloat162 *>(&final_u2.y) = res_high;

                    __stcs(reinterpret_cast<uint2 *>(&out_ptr[out_offset + w * 4]), final_u2);
                }
            }
        }
    }
}

template __global__ void varlen_embeddings_fused_kernel<__nv_bfloat16>(
    __nv_bfloat16 * __restrict__, const void * __restrict__, const float * __restrict__, const uint32_t * __restrict__, const int32_t * __restrict__, const int32_t * __restrict__, int32_t * __restrict__, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, bool);

template __global__ void varlen_embeddings_fused_kernel<__nv_fp8_e4m3>(
    __nv_bfloat16 * __restrict__, const void * __restrict__, const float * __restrict__, const uint32_t * __restrict__, const int32_t * __restrict__, const int32_t * __restrict__, int32_t * __restrict__, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, bool);

template __global__ void varlen_embeddings_fused_kernel<__nv_fp4_e2m1>(
    __nv_bfloat16 * __restrict__, const void * __restrict__, const float * __restrict__, const uint32_t * __restrict__, const int32_t * __restrict__, const int32_t * __restrict__, int32_t * __restrict__, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, bool);
