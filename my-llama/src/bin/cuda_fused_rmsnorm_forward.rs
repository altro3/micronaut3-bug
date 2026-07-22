use std::ffi::c_void;
use std::ptr;
use std::time::Instant;

use cuda_runtime::data_types::DataType;
use cuda_runtime::input_stage::fused_rmsnorm_forward;
use cuda_runtime::{
    CudaBuffer, device_synchronize, event_create, event_destroy, event_elapsed_time, event_record, event_synchronize, get_last_error,
    stream_create_with_flags, stream_destroy,
};
use my_llama::test_utils::{bf16_bits_to_f32, emu_fp4_e2m1_to_f32, emu_fp8_e4m3_to_f32, f32_to_bf16_bits};

fn run_rmsnorm_test(data_type: DataType, type_name: &str) {
    println!("\n=== RMSNORM ТЕЛЕМЕТРИЯ ФОРМАТА: {} ===", type_name);

    let hidden_size = 8192;
    let total_tokens = 4096;
    let threads_per_block = 256;
    let epsilon = 1e-5f32;

    let total_elements = (total_tokens * hidden_size) as usize;

    let input_bytes = match data_type {
        DataType::BF16 => total_elements * 2,
        DataType::FP8 => total_elements,
        DataType::FP4 => total_elements / 2,
    };

    let scale_elements = total_elements / 32;

    let h_input = vec![0x3Cu8; input_bytes];
    let mut h_gamma = vec![0u16; hidden_size as usize];
    for (i, item) in h_gamma.iter_mut().enumerate() {
        *item = f32_to_bf16_bits(1.0f32 + 0.001f32 * (i % 5) as f32);
    }
    let h_scales = vec![1.25f32; scale_elements];
    let mut h_output = vec![0u16; total_elements];

    let d_out = CudaBuffer::alloc(total_elements * 2);
    let d_input = CudaBuffer::alloc(input_bytes);
    let d_gamma = CudaBuffer::alloc((hidden_size * 2) as usize);
    let d_scales = CudaBuffer::alloc(scale_elements * 4);

    unsafe {
        d_input.copy_to_device(h_input.as_ptr() as *const c_void, input_bytes);
        d_gamma.copy_to_device(h_gamma.as_ptr() as *const c_void, (hidden_size * 2) as usize);
        d_scales.copy_to_device(h_scales.as_ptr() as *const c_void, scale_elements * 4);
    }

    let stream = stream_create_with_flags(0x01);

    unsafe {
        fused_rmsnorm_forward(
            d_out.ptr,
            d_input.ptr,
            d_gamma.ptr,
            d_scales.ptr as *const f32,
            epsilon,
            total_tokens,
            hidden_size,
            data_type,
            threads_per_block,
            stream,
        );
    }

    assert_eq!(device_synchronize(), 0);
    assert_eq!(get_last_error(), 0);

    const WARMUP: usize = 20;
    const ITERS: usize = 1000;

    for _ in 0..WARMUP {
        unsafe {
            fused_rmsnorm_forward(
                d_out.ptr,
                d_input.ptr,
                d_gamma.ptr,
                d_scales.ptr as *const f32,
                epsilon,
                total_tokens,
                hidden_size,
                data_type,
                threads_per_block,
                stream,
            );
        }
    }

    let mut start_events = vec![ptr::null_mut(); ITERS];
    let mut end_events = vec![ptr::null_mut(); ITERS];
    for item in start_events.iter_mut() {
        *item = event_create();
    }
    for item in end_events.iter_mut() {
        *item = event_create();
    }

    let start_host = Instant::now();
    for i in 0..ITERS {
        unsafe {
            event_record(start_events[i], stream);
            fused_rmsnorm_forward(
                d_out.ptr,
                d_input.ptr,
                d_gamma.ptr,
                d_scales.ptr as *const f32,
                epsilon,
                total_tokens,
                hidden_size,
                data_type,
                threads_per_block,
                stream,
            );
            event_record(end_events[i], stream);
        }
    }

    unsafe {
        event_synchronize(*end_events.last().unwrap());
    }
    let total_host_time = start_host.elapsed();

    let mut bandwidths: Vec<f64> = Vec::with_capacity(ITERS);
    let mut total_gpu_ms = 0.0_f32;

    let mut bytes_processed = (input_bytes as u64) + (hidden_size as u64 * 2) + (total_elements as u64 * 2);
    if data_type == DataType::FP4 {
        bytes_processed += (scale_elements as u64) * 4;
    }

    for i in 0..ITERS {
        let ms = unsafe { event_elapsed_time(start_events[i], end_events[i]) };
        total_gpu_ms += ms;
        bandwidths.push((bytes_processed as f64 / 1e9) / ((ms / 1000.0) as f64));
    }
    bandwidths.sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!("Wall Time:       {:.2} сек", total_host_time.as_secs_f32());
    println!("🚀 MAX ПСП RMS:  {:.2} ГБ/сек", bandwidths[ITERS - 1]);
    println!(
        "📈 AVG ПСП RMS:  {:.2} ГБ/сек",
        (bytes_processed as f64 * ITERS as f64 / 1e9) / (total_gpu_ms as f64 / 1000.0)
    );
    println!("⚠️ P95 ПСП RMS:  {:.2} ГБ/сек", bandwidths[(ITERS as f64 * 0.05) as usize]);

    unsafe {
        d_out.copy_to_host(h_output.as_mut_ptr() as *mut c_void, total_elements * 2);
    }
    println!("--- ЧЕСТНАЯ МАТЕМАТИЧЕСКАЯ ВАЛИДАЦИЯ RMSNORM ---");
    let mut math_errors = 0;
    let allowed_tolerance = 1e-2f32;

    for tok in 0..total_tokens as usize {
        let mut sum_sq = 0.0f32;
        let row_offset_bf16 = tok * hidden_size as usize;

        for f in 0..hidden_size as usize {
            let val = match data_type {
                DataType::BF16 => bf16_bits_to_f32(u16::from_le_bytes([
                    h_input[(row_offset_bf16 + f) * 2],
                    h_input[(row_offset_bf16 + f) * 2 + 1],
                ])),
                DataType::FP8 => emu_fp8_e4m3_to_f32(h_input[row_offset_bf16 + f]),
                DataType::FP4 => emu_fp4_e2m1_to_f32(h_input[tok * (hidden_size as usize / 2) + f / 2], f),
            };
            sum_sq += val * val;
        }

        let inv_rms = 1.0f32 / ((sum_sq / hidden_size as f32) + epsilon).sqrt();

        for f in 0..hidden_size as usize {
            let actual = bf16_bits_to_f32(h_output[row_offset_bf16 + f]);
            let inp = match data_type {
                DataType::BF16 => bf16_bits_to_f32(u16::from_le_bytes([
                    h_input[(row_offset_bf16 + f) * 2],
                    h_input[(row_offset_bf16 + f) * 2 + 1],
                ])),
                DataType::FP8 => emu_fp8_e4m3_to_f32(h_input[row_offset_bf16 + f]),
                DataType::FP4 => emu_fp4_e2m1_to_f32(h_input[tok * (hidden_size as usize / 2) + f / 2], f),
            };
            let g = bf16_bits_to_f32(h_gamma[f]);

            let expected = if data_type == DataType::FP4 {
                let scale_idx = (tok * hidden_size as usize + f) / 32;
                let scale = h_scales[scale_idx];
                inp * scale * inv_rms * g
            } else {
                inp * inv_rms * g
            };

            let diff = (actual - expected).abs();
            if diff > allowed_tolerance {
                if math_errors < 5 {
                    println!("Mismatch at token {}, feat {}: GPU={}, CPU={}, Diff={}", tok, f, actual, expected, diff);
                }
                math_errors += 1;
            }
        }
    }

    println!("Количество неверных элементов RMSNorm: {}", math_errors);
    assert_eq!(math_errors, 0, "Критическая ошибка математики в RMSNorm!");
    println!("✅ ВАЛИДАЦИЯ RMSNORM ПРОЙДЕНА ДЛЯ {}", type_name);

    unsafe {
        for i in 0..ITERS {
            event_destroy(start_events[i]);
            event_destroy(end_events[i]);
        }
        stream_destroy(stream);
    }
}

fn main() {
    run_rmsnorm_test(DataType::BF16, "BF16");
    run_rmsnorm_test(DataType::FP8, "FP8");
    run_rmsnorm_test(DataType::FP4, "FP4");
}
