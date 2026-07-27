#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>
#include "dequantize_fusion_mma_gguf_q4_k.cuh"
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder_decl.hpp>
#include <cutlass/gemm/kernel/default_gemm_universal.h>
#include <cutlass/util/packed_stride.hpp>

namespace cutlass::epilogue::collective {
    struct EpilogueTileAuto;
}

using namespace cute;

__global__ void __launch_bounds__(256) fused_gguf_to_nvfp4_blockscaled_kernel(
    float_e2m1_t * __restrict__ output_B,
    float_ue4m3_t * __restrict__ output_SFB,
    const BlockQ4K * __restrict__ input_B_quant,
    const int32_t N,
    const int32_t K
) {
    const int32_t global_n = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_n >= N) return;

    const int32_t num_blocks_k = K / 256;

    for (int32_t b_k = 0; b_k < num_blocks_k; ++b_k) {
        const int32_t block_idx = global_n * num_blocks_k + b_k;
        const BlockQ4K &block = input_B_quant[block_idx];

        const float d_val = __bfloat162float(block.d);
        const float dmin_val = __bfloat162float(block.dmin);

        float block_max_abs = 1e-5f;
        float unpacked_vals[128];

#pragma unroll 4
        for (int32_t elem_idx = 0; elem_idx < 128; ++elem_idx) {
            const int32_t sub_block_idx = elem_idx / 32;
            const int32_t sub_elem_idx = elem_idx % 32;

            const int32_t bit_offset_sc = sub_block_idx * 6;
            const int32_t byte_offset_sc = bit_offset_sc / 8;
            const int32_t bit_shift_sc = bit_offset_sc % 8;
            uint32_t val_sc = block.scales[byte_offset_sc] | (block.scales[byte_offset_sc + 1] << 8);
            if (byte_offset_sc + 2 < 12) {
                val_sc |= block.scales[byte_offset_sc + 2] << 16;
            }
            const uint8_t sc = (val_sc >> bit_shift_sc) & 0x3F;

            const int32_t bit_offset_min = (sub_block_idx + 8) * 6;
            const int32_t byte_offset_min = bit_offset_min / 8;
            const int32_t bit_shift_min = bit_offset_min % 8;
            uint32_t val_min = block.scales[byte_offset_min] | (block.scales[byte_offset_min + 1] << 8);
            if (byte_offset_min + 2 < 12) {
                val_min |= block.scales[byte_offset_min + 2] << 16;
            }
            const uint8_t min_sc = (val_min >> bit_shift_min) & 0x3F;

            const uint8_t qs_byte = block.qs[sub_block_idx * 16 + sub_elem_idx % 16];
            const uint8_t raw_q = sub_elem_idx < 16 ? qs_byte & 0x0F : qs_byte >> 4;

            const float out_fp32 = d_val * static_cast<float>(sc) * static_cast<float>(raw_q) - dmin_val * static_cast<float>(min_sc);
            unpacked_vals[elem_idx] = out_fp32;

            const float abs_val = fabsf(out_fp32);
            if (abs_val > block_max_abs) {
                block_max_abs = abs_val;
            }
        }

        const float sf_identity = block_max_abs / 6.0f;
        const float inv_sf = 1.0f / sf_identity;

        __half h_sf = __float2half(sf_identity);
        const __half_raw h_raw_sf = *reinterpret_cast<__half_raw *>(&h_sf);

        const int32_t global_sf_idx = global_n * num_blocks_k + b_k;
        output_SFB[global_sf_idx] = float_ue4m3_t(__nv_cvt_halfraw_to_fp8(h_raw_sf, __NV_NOSAT, __NV_E4M3));

        const int32_t global_weight_base_idx = global_n * K + b_k * 128;
        const auto fp4_out_ptr = reinterpret_cast<uint32_t *>(&output_B[global_weight_base_idx / 2]);

#pragma unroll 4
        for (int32_t pack_idx = 0; pack_idx < 16; ++pack_idx) {
            uint32_t packed_val = 0;
#pragma unroll
            for (int32_t v = 0; v < 8; ++v) {
                const int32_t elem_idx = pack_idx * 8 + v;
                const float scaled_val = unpacked_vals[elem_idx] * inv_sf;

                __half h_val = __float2half(scaled_val);
                const __half_raw h_raw = *reinterpret_cast<__half_raw *>(&h_val);

                const uint32_t fp4_bits = __nv_cvt_halfraw_to_fp4(h_raw, __NV_E2M1, cudaRoundNearest) & 0x0F;
                packed_val |= fp4_bits << (v * 4);
            }
            fp4_out_ptr[pack_idx] = packed_val;
        }
    }
}

void run_fused_gemm_gguf_blackwell_fp4(
    bfloat16_t *output_D,
    const float_e2m1_t *input_A,
    const float_ue4m3_t *input_SFA,
    const BlockQ4K *input_B_gguf,
    int32_t M, int32_t N, int32_t K,
    cudaStream_t stream
) {
    float_e2m1_t *dev_B_nvfp4 = nullptr;
    float_ue4m3_t *dev_SFB_nvfp4 = nullptr;

    cudaMalloc(reinterpret_cast<void **>(&dev_B_nvfp4), N * K * sizeof(uint8_t) / 2);
    cudaMalloc(reinterpret_cast<void **>(&dev_SFB_nvfp4), N * (K / 128) * sizeof(uint8_t));

    dim3 conversion_block(256);
    dim3 conversion_grid((N + 255) / 256);

    fused_gguf_to_nvfp4_blockscaled_kernel<<<conversion_grid, conversion_block, 0, stream>>>(
        dev_B_nvfp4, dev_SFB_nvfp4, input_B_gguf, N, K
    );

    using ElementA = float_e2m1_t;
    using ElementSFA = float_ue4m3_t;
    using LayoutATag = cutlass::layout::RowMajor;
    constexpr int AlignmentA = 32;

    using ElementB = float_e2m1_t;
    using ElementSFB = float_ue4m3_t;
    using LayoutBTag = cutlass::layout::ColumnMajor;
    constexpr int AlignmentB = 32;

    using ElementD = bfloat16_t;
    using ElementC = bfloat16_t;
    using LayoutCTag = cutlass::layout::RowMajor;
    using LayoutDTag = cutlass::layout::RowMajor;
    constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
    constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

    using ElementAccumulator = float;
    using ArchTag = cutlass::arch::Sm103;
    using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

    using MmaTileShape2Sm = Shape<_256, _256, Int<768> >;
    using ClusterShape = Shape<int, int, _1>;

    using CollectiveEpilogue2Sm = cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        MmaTileShape2Sm, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutCTag, AlignmentC,
        ElementD, LayoutDTag, AlignmentD,
        cutlass::epilogue::NoSmemWarpSpecialized2Sm
    >::CollectiveOp;

    using CollectiveMainloop2Sm = cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        tuple<ElementA, ElementSFA, ElementSFA>, LayoutATag, AlignmentA,
        tuple<ElementB, ElementSFB, ElementSFB>, LayoutBTag, AlignmentB,
        ElementAccumulator,
        MmaTileShape2Sm, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(CollectiveEpilogue2Sm::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecialized2SmBlockScaledMxNvf4UltraVs16Sm103
    >::CollectiveOp;

    using GemmKernelInstance = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop2Sm,
        CollectiveEpilogue2Sm
    >;
    using GemmInstance = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelInstance>;

    GemmInstance gemm;
    GemmInstance::Arguments arguments{
        .mode = cutlass::gemm::GemmUniversalMode::kGemm,
        .problem_shape = {M, N, K, 1},
        .mainloop = {
            .ptr_A = input_A, .dA = cutlass::make_cute_packed_stride(GemmInstance::GemmKernel::StrideA{}, {M, K, 1}),
            .ptr_B = dev_B_nvfp4, .dB = cutlass::make_cute_packed_stride(GemmInstance::GemmKernel::StrideB{}, {N, K, 1}),
            .ptr_SFA = input_SFA, .layout_SFA = GemmInstance::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1)),
            .ptr_SFB = dev_SFB_nvfp4, .layout_SFB = GemmInstance::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1))
        },
        .epilogue = {
            .thread = {.alpha = 1.0f, .beta = 0.0f},
            .ptr_C = nullptr, .dC = cutlass::make_cute_packed_stride(GemmInstance::GemmKernel::StrideC{}, {M, N, 1}),
            .ptr_D = output_D, .dD = cutlass::make_cute_packed_stride(GemmInstance::GemmKernel::StrideD{}, {M, N, 1})
        }
    };

    arguments.scheduler.max_swizzle_size = 0;
    arguments.hw_info.cluster_shape = dim3(2, 1, 1);
    arguments.hw_info.cluster_shape_fallback = dim3(2, 1, 1);

    const size_t workspace_size = GemmInstance::get_workspace_size(arguments);
    uint8_t *workspace = nullptr;
    cudaMalloc(&workspace, workspace_size);

    gemm.initialize(arguments, workspace, stream);
    gemm.run(stream);

    cudaFree(workspace);
    cudaFree(dev_B_nvfp4);
    cudaFree(dev_SFB_nvfp4);
}
