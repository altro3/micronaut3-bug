#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <type_traits>
#include <stdint.h>

#include "fused_rmsnorm_forward.cuh"

template<typename T>
__global__ void fused_rmsnorm_forward_kernel(
    __nv_bfloat16 * __restrict__ out,
    const void * __restrict__ input,
    const void * __restrict__ gamma,
    const float * __restrict__ gamma_scales,
    float epsilon,
    int32_t total_tokens,
    int32_t hidden_size
) {
    extern __shared__ float s_warp_sums[];

    const int32_t token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;

    const int32_t tid = threadIdx.x;
    const int32_t lane_id = tid % 32;
    const int32_t warp_id = tid / 32;

    const int64_t row_offset_bf16 = static_cast<int64_t>(token_idx) * hidden_size;
    __nv_bfloat16 *const row_out = out + row_offset_bf16;

    float thread_sum_sq = 0.0f;

    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        const __nv_bfloat16 *const row_in = static_cast<const __nv_bfloat16 *>(input) + row_offset_bf16;
        const int32_t stride = blockDim.x * 8;

        for (int32_t i = tid * 8; i < hidden_size; i += stride) {
            uint4 in_v4 = __ldcs(reinterpret_cast<const uint4 *>(&row_in[i]));
            auto h2_ptr = reinterpret_cast<const __nv_bfloat162 *>(&in_v4);
#pragma unroll
            for (int32_t j = 0; j < 4; ++j) {
                float2 f2 = __bfloat1622float2(h2_ptr[j]);
                thread_sum_sq += f2.x * f2.x + f2.y * f2.y;
            }
        }
    } else if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
        const int64_t row_offset_fp8 = static_cast<int64_t>(token_idx) * hidden_size;
        const __nv_fp8_e4m3 *const row_in = static_cast<const __nv_fp8_e4m3 *>(input) + row_offset_fp8;
        const int32_t stride = blockDim.x * 16;

        for (int32_t i = tid * 16; i < hidden_size; i += stride) {
            uint4 in_v4 = __ldcs(reinterpret_cast<const uint4 *>(&row_in[i]));
            auto packed = reinterpret_cast<const uint32_t *>(&in_v4);
#pragma unroll
            for (int32_t j = 0; j < 4; ++j) {
                uint32_t val32 = packed[j];
                float2 f2_low = __half22float2(__nv_cvt_fp8x2_to_halfraw2(static_cast<uint16_t>(val32 & 0xFFFF), __NV_E4M3));
                float2 f2_high = __half22float2(__nv_cvt_fp8x2_to_halfraw2(static_cast<uint16_t>((val32 >> 16) & 0xFFFF), __NV_E4M3));
                thread_sum_sq += f2_low.x * f2_low.x + f2_low.y * f2_low.y + f2_high.x * f2_high.x + f2_high.y * f2_high.y;
            }
        }
    } else if constexpr (std::is_same_v<T, __nv_fp4_e2m1>) {
        const int64_t row_offset_fp4 = static_cast<int64_t>(token_idx) * (hidden_size / 8);
        const uint32_t *const row_in = static_cast<const uint32_t *>(input) + row_offset_fp4;

        for (int32_t i = tid; i < hidden_size / 8; i += blockDim.x) {
            uint32_t packed_val32 = __ldcs(&row_in[i]);
#pragma unroll
            for (int32_t byte_idx = 0; byte_idx < 4; ++byte_idx) {
                uint8_t byte = static_cast<uint8_t>((packed_val32 >> (byte_idx * 8)) & 0xFF);
                __half2_raw raw_h2 = __nv_cvt_fp4x2_to_halfraw2(byte, __NV_E2M1);
                float2 f2 = __half22float2(*reinterpret_cast<__half2 *>(&raw_h2));
                thread_sum_sq += f2.x * f2.x + f2.y * f2.y;
            }
        }
    }

#pragma unroll
    for (int32_t offset = 16; offset > 0; offset /= 2) {
        thread_sum_sq += __shfl_xor_sync(0xFFFFFFFF, thread_sum_sq, offset);
    }

    if (lane_id == 0) {
        s_warp_sums[warp_id] = thread_sum_sq;
    }
    __syncthreads();

    const int32_t num_warps = blockDim.x / 32;
    if (warp_id == 0) {
        float block_sum_sq = 0.0f;
        block_sum_sq = lane_id < num_warps ? s_warp_sums[lane_id] : 0.0f;
#pragma unroll
        for (int32_t offset = 16; offset > 0; offset /= 2) {
            block_sum_sq += __shfl_xor_sync(0xFFFFFFFF, block_sum_sq, offset);
        }
        if (lane_id == 0) {
            s_warp_sums[0] = rsqrtf(block_sum_sq / static_cast<float>(hidden_size) + epsilon);
        }
    }
    __syncthreads();

    const float inv_rms = s_warp_sums[0];
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        const __nv_bfloat16 *const row_in = static_cast<const __nv_bfloat16 *>(input) + row_offset_bf16;
        const auto g_ptr = static_cast<const __nv_bfloat16 *>(gamma);
        const int32_t stride = blockDim.x * 8;

        for (int32_t i = tid * 8; i < hidden_size; i += stride) {
            uint4 in_v4 = __ldcs(reinterpret_cast<const uint4 *>(&row_in[i]));
            uint4 gamma_v4 = __ldcs(reinterpret_cast<const uint4 *>(&g_ptr[i]));

            auto h2_in = reinterpret_cast<const __nv_bfloat162 *>(&in_v4);
            auto h2_gamma = reinterpret_cast<const __nv_bfloat162 *>(&gamma_v4);
            uint4 out_v4;
            auto h2_out = reinterpret_cast<__nv_bfloat162 *>(&out_v4);

#pragma unroll
            for (int32_t j = 0; j < 4; ++j) {
                float2 f2_in = __bfloat1622float2(h2_in[j]);
                float2 f2_gamma = __bfloat1622float2(h2_gamma[j]);
                h2_out[j] = __floats2bfloat162_rn(f2_in.x * inv_rms * f2_gamma.x, f2_in.y * inv_rms * f2_gamma.y);
            }
            __stcs(reinterpret_cast<uint4 *>(&row_out[i]), out_v4);
        }
    } else if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
        const __nv_fp8_e4m3 *const row_in = static_cast<const __nv_fp8_e4m3 *>(input) + row_offset_bf16;
        const auto g_ptr = static_cast<const __nv_bfloat16 *>(gamma);
        const int32_t stride = blockDim.x * 8;

        for (int32_t i = tid * 8; i < hidden_size; i += stride) {
            uint2 in_u2 = __ldcs(reinterpret_cast<const uint2 *>(&row_in[i]));
            uint4 gamma_v4 = __ldcs(reinterpret_cast<const uint4 *>(&g_ptr[i]));
            auto packed_vals = reinterpret_cast<const uint32_t *>(&in_u2);
            auto h2_gamma = reinterpret_cast<const __nv_bfloat162 *>(&gamma_v4);

            uint4 out_v4;
            auto h2_out = reinterpret_cast<__nv_bfloat162 *>(&out_v4);

#pragma unroll
            for (int32_t j = 0; j < 2; ++j) {
                uint32_t val32 = packed_vals[j];
                float2 f2_low = __half22float2(__nv_cvt_fp8x2_to_halfraw2(static_cast<uint16_t>(val32 & 0xFFFF), __NV_E4M3));
                float2 f2_high = __half22float2(__nv_cvt_fp8x2_to_halfraw2(static_cast<uint16_t>((val32 >> 16) & 0xFFFF), __NV_E4M3));

                float2 g0 = __bfloat1622float2(h2_gamma[j * 2]);
                float2 g1 = __bfloat1622float2(h2_gamma[j * 2 + 1]);

                h2_out[j * 2] = __floats2bfloat162_rn(f2_low.x * inv_rms * g0.x, f2_low.y * inv_rms * g0.y);
                h2_out[j * 2 + 1] = __floats2bfloat162_rn(f2_high.x * inv_rms * g1.x, f2_high.y * inv_rms * g1.y);
            }
            __stcs(reinterpret_cast<uint4 *>(&row_out[i]), out_v4);
        }
    } else if constexpr (std::is_same_v<T, __nv_fp4_e2m1>) {
        const uint32_t *const row_in = static_cast<const uint32_t *>(input) + (static_cast<int64_t>(token_idx) * (hidden_size / 8));
        const auto g_ptr = static_cast<const __nv_bfloat16 *>(gamma);
        const int32_t stride = blockDim.x;

        for (int32_t i = tid; i < hidden_size / 8; i += stride) {
            uint32_t packed_val32 = __ldcs(&row_in[i]);
            const int32_t out_base = i * 8;

            uint4 gamma_v4 = __ldcs(reinterpret_cast<const uint4 *>(&g_ptr[out_base]));
            auto h2_gamma = reinterpret_cast<const __nv_bfloat162 *>(&gamma_v4);

            uint4 out_v4;
            auto h2_out = reinterpret_cast<__nv_bfloat162 *>(&out_v4);

#pragma unroll
            for (int32_t byte_idx = 0; byte_idx < 4; ++byte_idx) {
                uint8_t byte = static_cast<uint8_t>((packed_val32 >> (byte_idx * 8)) & 0xFF);
                __half2_raw raw_h2 = __nv_cvt_fp4x2_to_halfraw2(byte, __NV_E2M1);
                float2 f2 = __half22float2(*reinterpret_cast<__half2 *>(&raw_h2));

                const int32_t scale_group = (out_base + byte_idx * 2) / 32;
                const float scale = gamma_scales[scale_group];

                float2 g = __bfloat1622float2(h2_gamma[byte_idx]);

                h2_out[byte_idx] = __floats2bfloat162_rn(
                    f2.x * scale * inv_rms * g.x,
                    f2.y * scale * inv_rms * g.y
                );
            }

            __stcs(reinterpret_cast<uint4 *>(&row_out[out_base]), out_v4);
        }
    }
}

template __global__ void fused_rmsnorm_forward_kernel<__nv_bfloat16>(__nv_bfloat16 * __restrict__, const void * __restrict__, const void * __restrict__, const float * __restrict__, float, int32_t, int32_t);

template __global__ void fused_rmsnorm_forward_kernel<__nv_fp8_e4m3>(__nv_bfloat16 * __restrict__, const void * __restrict__, const void * __restrict__, const float * __restrict__, float, int32_t, int32_t);

template __global__ void fused_rmsnorm_forward_kernel<__nv_fp4_e2m1>(__nv_bfloat16 * __restrict__, const void * __restrict__, const void * __restrict__, const float * __restrict__, float, int32_t, int32_t);
