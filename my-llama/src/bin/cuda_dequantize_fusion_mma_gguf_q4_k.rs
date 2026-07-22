use std::ffi::c_void;
use std::ptr;
use std::time::Instant;

use cuda_runtime::data_types::{DataType, QuantType};
use cuda_runtime::prefill_stage::prefill_linear;
use cuda_runtime::{
    CudaBuffer, device_synchronize, event_create, event_destroy, event_elapsed_time, event_record, event_synchronize, stream_create_with_flags,
    stream_destroy,
};
use my_llama::test_utils::{bf16_bits_to_f32, emu_fp4_e2m1_to_f32, emu_fp8_e4m3_to_f32, f32_to_bf16_bits};

#[repr(C, align(4))]
#[derive(Clone, Copy)]
struct BlockQ4K {
    d: u16,
    dmin: u16,
    scales: [u8; 12],
    qs: [u8; 128],
}

fn run_dequantize_mma_test(data_type: DataType) {
    let type_name = format!("{:?}", data_type);
    println!("\n=== GGUF Q4_K MMA ТЕЛЕМЕТРИЯ ФОРМАТА АКТИВАЦИЙ: {} ===", type_name);

    let batch_size_or_tokens = 2048;
    let hidden_units_out = 4096;
    let hidden_units_in = 8192;

    let total_elements_a = (batch_size_or_tokens * hidden_units_in) as usize;
    let total_elements_b = (hidden_units_out * hidden_units_in) as usize;
    let total_elements_c = (batch_size_or_tokens * hidden_units_out) as usize;

    let input_a_bytes = match data_type {
        DataType::BF16 => total_elements_a * 2,
        DataType::FP8 => total_elements_a,
        DataType::FP4 => total_elements_a / 2,
    };

    let output_bytes_actual = match data_type {
        DataType::BF16 => total_elements_c * 2,
        DataType::FP8 => total_elements_c,
        DataType::FP4 => total_elements_c / 2,
    };

    let num_gguf_blocks = total_elements_b / 256;
    let mut h_quant_weights = vec![BlockQ4K { d: 0, dmin: 0, scales: [0; 12], qs: [0; 128] }; num_gguf_blocks];

    for (i, block) in h_quant_weights.iter_mut().enumerate() {
        block.d = f32_to_bf16_bits(0.01f32 * (i % 3 + 1) as f32);
        block.dmin = f32_to_bf16_bits(0.002f32 * (i % 2 + 1) as f32);
        for s in 0..12 {
            block.scales[s] = (32 + (s % 4) * 8) as u8;
        }
        for q in 0..128 {
            block.qs[q] = (q % 256) as u8;
        }
    }

    let h_input_a = vec![0x3Cu8; input_a_bytes];
    let mut h_output_c = vec![0u8; output_bytes_actual];

    let d_output_c = CudaBuffer::alloc(output_bytes_actual);
    let d_input_a = CudaBuffer::alloc(input_a_bytes);
    let d_quant_weights = CudaBuffer::alloc(num_gguf_blocks * size_of::<BlockQ4K>());

    unsafe {
        d_input_a.copy_to_device(h_input_a.as_ptr() as *const c_void, input_a_bytes);
        d_quant_weights.copy_to_device(h_quant_weights.as_ptr() as *const c_void, num_gguf_blocks * size_of::<BlockQ4K>());
    }

    let stream = stream_create_with_flags(0x01);

    const WARMUP: usize = 10;
    const ITERS: usize = 100;

    for _ in 0..WARMUP {
        unsafe {
            prefill_linear(
                d_output_c.ptr,
                d_input_a.ptr,
                d_quant_weights.ptr,
                batch_size_or_tokens,
                hidden_units_out,
                hidden_units_in,
                data_type,
                QuantType::GgufQ4K,
                stream,
            );
        }
    }
    assert_eq!(device_synchronize(), 0);

    let mut start_events = vec![ptr::null_mut(); ITERS];
    let mut end_events = vec![ptr::null_mut(); ITERS];
    for i in 0..ITERS {
        start_events[i] = event_create();
        end_events[i] = event_create();
    }

    let start_host = Instant::now();
    for i in 0..ITERS {
        unsafe {
            event_record(start_events[i], stream);
            prefill_linear(
                d_output_c.ptr,
                d_input_a.ptr,
                d_quant_weights.ptr,
                batch_size_or_tokens,
                hidden_units_out,
                hidden_units_in,
                data_type,
                QuantType::GgufQ4K,
                stream,
            );
            event_record(end_events[i], stream);
        }
    }

    unsafe {
        event_synchronize(*end_events.last().unwrap());
    }
    let total_host_time = start_host.elapsed();

    let mut total_gpu_ms = 0.0_f32;
    for i in 0..ITERS {
        total_gpu_ms += unsafe { event_elapsed_time(start_events[i], end_events[i]) };
    }

    let avg_gpu_ms = total_gpu_ms / (ITERS as f32);
    let avg_host_ms = (total_host_time.as_secs_f32() * 1000.0f32) / (ITERS as f32);

    let total_ops = 2.0 * (batch_size_or_tokens as f64) * (hidden_units_out as f64) * (hidden_units_in as f64);
    let tflops = if avg_gpu_ms > 0.0 { (total_ops * 1e-12) / (avg_gpu_ms as f64 * 1e-3) } else { 0.0 };

    let weight_bytes_actual = num_gguf_blocks * size_of::<BlockQ4K>();
    let algorithmic_bytes = input_a_bytes as u64 + weight_bytes_actual as u64 + output_bytes_actual as u64;
    let bandwidth_gbps = if avg_gpu_ms > 0.0 {
        (algorithmic_bytes as f64 * 1e-9) / (avg_gpu_ms as f64 * 1e-3)
    } else {
        0.0
    };

    println!("  - Среднее время выполнения на GPU: {:.3} ms", avg_gpu_ms);
    println!("  - Среднее время отправки с хоста:  {:.3} ms", avg_host_ms);
    println!("  - Алгоритмическая мощность:        {:.2} TFLOPS", tflops);
    println!("  - Полезная ПСП ядра:                {:.2} GB/s", bandwidth_gbps);
    println!("[CPU ДЕБАГ] Физический адрес h_output_c: {:p}", h_output_c.as_ptr());
    println!("[CPU ДЕБАГ] Первые 8 байт из h_output_c, которые вернул GPU: {:?}", &h_output_c[0..8]);
    println!(
        "[CPU ДЕБАГ] Значение первого элемента по формуле теста (actual): {}",
        match data_type {
            DataType::FP4 => emu_fp4_e2m1_to_f32(h_output_c[0], 0),
            _ => 0.0,
        }
    );

    unsafe {
        let sync_res = device_synchronize();
        if sync_res != 0 {
            println!("[RUST ERROR] CUDA device synchronization failed with code: {}", sync_res);
        }

        d_output_c.copy_to_host(h_output_c.as_mut_ptr() as *mut c_void, output_bytes_actual);
    }
    println!("--- ЧЕСТНАЯ МАТЕМАТИЧЕСКАЯ ВАЛИДАЦИЯ СЛИТОГО ЯДРА ДЕКВАНТОВАНИЯ ---");
    let mut math_errors = 0;
    let allowed_tolerance = 1e-1f32;

    for row in 0..std::cmp::min(batch_size_or_tokens as usize, 4) {
        for col in 0..std::cmp::min(hidden_units_out as usize, 16) {
            let mut accum = 0.0f32;

            for contr in 0..hidden_units_in as usize {
                let input_val_fp32 = match data_type {
                    DataType::BF16 => bf16_bits_to_f32(u16::from_le_bytes([
                        h_input_a[(row * hidden_units_in as usize + contr) * 2],
                        h_input_a[(row * hidden_units_in as usize + contr) * 2 + 1],
                    ])),
                    DataType::FP8 => emu_fp8_e4m3_to_f32(h_input_a[row * hidden_units_in as usize + contr]),
                    DataType::FP4 => emu_fp4_e2m1_to_f32(h_input_a[(row * hidden_units_in as usize + contr) / 2], contr),
                };

                let input_bf16_bits = f32_to_bf16_bits(input_val_fp32);
                let input_val = bf16_bits_to_f32(input_bf16_bits);

                let weight_element_idx = col * hidden_units_in as usize + contr;
                let block_idx = weight_element_idx / 256;
                let elem_in_block = weight_element_idx % 256;

                let block = &h_quant_weights[block_idx];
                let d_val = bf16_bits_to_f32(block.d);
                let dmin_val = bf16_bits_to_f32(block.dmin);

                let j = elem_in_block / 64;
                let il = (elem_in_block % 64) / 16;
                let pair_idx = elem_in_block % 16;

                let j_mod = j % 2;

                let s_low = block.scales[j_mod * 4 + il];
                let s_high = block.scales[8 + j_mod * 2 + il / 2];

                let sc = s_low & 63;
                let min_sc = s_high & 63;

                let d_super = d_val * sc as f32;
                let m_super = dmin_val * min_sc as f32;

                let byte_idx = j * 32 + il * 4 + pair_idx / 2;
                let packed_byte = block.qs[byte_idx];
                let raw_q = if pair_idx % 2 == 0 { packed_byte & 0x0F } else { packed_byte >> 4 };

                let weight_val_fp32 = d_super * raw_q as f32 - m_super;

                let weight_bf16_bits = f32_to_bf16_bits(weight_val_fp32);
                let weight_val = bf16_bits_to_f32(weight_bf16_bits);

                accum += input_val * weight_val;
            }

            let accum_bf16_bits = f32_to_bf16_bits(accum);
            accum = bf16_bits_to_f32(accum_bf16_bits);

            let linear_idx = row * hidden_units_out as usize + col;
            let actual = match data_type {
                DataType::BF16 => {
                    let b0 = h_output_c[linear_idx * 2];
                    let b1 = h_output_c[linear_idx * 2 + 1];
                    bf16_bits_to_f32(u16::from_le_bytes([b0, b1]))
                },
                DataType::FP8 => emu_fp8_e4m3_to_f32(h_output_c[linear_idx]),
                DataType::FP4 => emu_fp4_e2m1_to_f32(h_output_c[linear_idx / 2], linear_idx),
            };

            let diff = (actual - accum).abs();
            if diff > allowed_tolerance {
                if math_errors < 5 {
                    println!(
                        "Расхождение в строке {}, столбец {}: GPU={}, CPU={}, Разница={}",
                        row, col, actual, accum, diff
                    );
                }
                math_errors += 1;
            }
        }
    }

    println!("Количество неверных элементов в выборке: {}", math_errors);
    assert_eq!(math_errors, 0, "Критическая ошибка математики ядра деквантования весов!");
    println!("✅ ВАЛИДАЦИЯ ДЕКВАНТОВАНИЯ GGUF Q4_K ПРОЙДЕНА УСПЕШНО ДЛЯ ФОРМАТА {}", type_name);

    unsafe {
        for i in 0..ITERS {
            event_destroy(start_events[i]);
            event_destroy(end_events[i]);
        }
        stream_destroy(stream);
    }
}

fn main() {
    println!("=== ЗАПУСК ПРОМЫШЛЕННОГО ТЕСТА СЛИТОГО MMA-ЯДРА DEQUANTIZE_GGUF_Q4_K ===");
    run_dequantize_mma_test(DataType::BF16);
    run_dequantize_mma_test(DataType::FP8);
    run_dequantize_mma_test(DataType::FP4);
    println!("\n🚀 УСПЕХ! ВСЕ КОМБИНАЦИИ АКТИВАЦИЙ С ВЕСАМИ Q4_K СКОМПИЛИРОВАНЫ И ПРОШЛИ КОРРЕКТНУЮ СВЕРКУ!");
}
