#include <cuda_fp16.h>
#include <cuda_runtime.h>

struct __align__(4) BlockQ4K {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

__global__ void dequantize_q4_k_kernel_v3(
    float4 * __restrict__ output,
    const BlockQ4K * __restrict__ input,
    const int num_blocks
) {
    for (int block_idx = blockIdx.x; block_idx < num_blocks; block_idx += gridDim.x) {
        const BlockQ4K &block = input[block_idx];

        const float d_val = __half2float(block.d);
        const float dmin_val = __half2float(block.dmin);

        const int tid = threadIdx.x;
        const int j = tid / 4;
        const int il = tid % 4;

        const int is_high = j >> 2;
        const int j_mod = j & 3;

        const uint8_t s_low = block.scales[j_mod + is_high * 6];
        const uint8_t s_high = block.scales[j_mod + 4 + is_high * 2];

        const uint8_t sc = (s_low & 63 + is_high * 192) >> (is_high * 6) | (s_high & 63 - is_high * 47) << (2 + is_high * 2);
        const uint8_t min_sc = (block.scales[j_mod + 6 - is_high * 6] & 63 + is_high * 192) >> (is_high * 6) | (s_high & 12 + is_high * 228) >> (2 - is_high * 4);

        const float d_super = d_val * static_cast<float>(sc);
        const float m_super = dmin_val * static_cast<float>(min_sc);

        const int qs_offset = j * 16 + il * 4;
        const uint32_t bytes = *reinterpret_cast<const uint32_t *>(&block.qs[qs_offset]);

        const uint32_t b0 = bytes & 0xFF;
        const uint32_t b1 = bytes >> 8 & 0xFF;
        const uint32_t b2 = bytes >> 16 & 0xFF;
        const uint32_t b3 = bytes >> 24;

        float4 out_val1;
        float4 out_val2;

        out_val1.x = d_super * static_cast<float>(b0 & 0x0F) - m_super;
        out_val1.y = d_super * static_cast<float>(b1 & 0x0F) - m_super;
        out_val1.z = d_super * static_cast<float>(b2 & 0x0F) - m_super;
        out_val1.w = d_super * static_cast<float>(b3 & 0x0F) - m_super;

        out_val2.x = d_super * static_cast<float>(b0 >> 4) - m_super;
        out_val2.y = d_super * static_cast<float>(b1 >> 4) - m_super;
        out_val2.z = d_super * static_cast<float>(b2 >> 4) - m_super;
        out_val2.w = d_super * static_cast<float>(b3 >> 4) - m_super;

        float4 *const block_out = output + block_idx * 64;

        __stcs(&block_out[j * 16 + il], out_val1);
        __stcs(&block_out[j * 16 + il + 4], out_val2);
    }
}

extern "C" {
void launch_dequantize_q4_k(
    float *output,
    const void *input,
    const int num_elements,
    void *stream_ptr
) {
    const auto stream = static_cast<cudaStream_t>(stream_ptr);
    const int num_blocks = num_elements / 256;

    if (num_blocks > 0) {
        constexpr int threads = 32;
        const int blocks = min(2048, num_blocks);

        dequantize_q4_k_kernel_v3<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<float4 *>(output),
            static_cast<const BlockQ4K *>(input),
            num_blocks
        );
    }
}
}
