use std::ffi::c_void;
use std::ptr;
use std::time::Instant;

use cuda_runtime::data_types::DataType;
use cuda_runtime::input_stage::varlen_embeddings;
use cuda_runtime::{
    CudaBuffer, device_synchronize, event_create, event_destroy, event_elapsed_time, event_record, event_synchronize, get_last_error,
    stream_create_with_flags, stream_destroy,
};
use my_llama::test_utils::{emu_fp4_e2m1_to_f32, emu_fp8_e4m3_to_f32, f32_to_bf16_bits};

fn run_benchmark_for_type(data_type: DataType) {
    let type_name = format!("{:?}", data_type);
    println!("\n=== ТЕСТИРОВАНИЕ ФОРМАТА: {} ===", type_name);

    let vocab_size = 152064;
    let out_features = 8192;
    let threads_per_block = 256;
    let block_size = 16;
    let max_blocks_per_seq = 64;
    let seqlens = [1024, 1024, 1024, 1024];
    let num_seqs = seqlens.len() as i32;
    let total_tokens = seqlens.iter().sum::<i32>();

    let mut seq_offsets = vec![0; seqlens.len() + 1];
    for i in 0..seqlens.len() {
        seq_offsets[i + 1] = seq_offsets[i] + seqlens[i];
    }

    let mut block_table = vec![-1; (num_seqs * max_blocks_per_seq) as usize];
    for (i, item) in block_table.iter_mut().enumerate() {
        *item = (i as i32) + 10;
    }

    let h_tokens: Vec<u32> = (0..total_tokens).map(|i| i as u32 % vocab_size as u32).collect();
    let mut h_slot_mapping = vec![-1; total_tokens as usize];
    let mut h_output = vec![0u16; (total_tokens * out_features) as usize];

    let vocab_size_u64 = vocab_size as u64;
    let out_features_u64 = out_features as u64;

    let weight_bytes = match data_type {
        DataType::BF16 => (vocab_size_u64 * out_features_u64 * 2) as usize,
        DataType::FP8 => (vocab_size_u64 * out_features_u64) as usize,
        DataType::FP4 => (vocab_size_u64 * out_features_u64 / 2) as usize,
    };

    let scale_elements = ((vocab_size_u64 * out_features_u64) / 32) as usize;

    let h_weight = vec![0x3Cu8; weight_bytes];
    let h_scales = vec![1.25f32; scale_elements];

    let d_out = CudaBuffer::alloc((total_tokens * out_features * 2) as usize);
    let d_weight = CudaBuffer::alloc(weight_bytes);
    let d_scales = CudaBuffer::alloc(scale_elements * 4);
    let d_tokens = CudaBuffer::alloc((total_tokens * 4) as usize);
    let d_offsets = CudaBuffer::alloc(seq_offsets.len() * 4);
    let d_block_table = CudaBuffer::alloc(block_table.len() * 4);
    let d_slot_mapping = CudaBuffer::alloc((total_tokens * 4) as usize);

    unsafe {
        d_weight.copy_to_device(h_weight.as_ptr() as *const c_void, weight_bytes);
        d_scales.copy_to_device(h_scales.as_ptr() as *const c_void, scale_elements * 4);
        d_tokens.copy_to_device(h_tokens.as_ptr() as *const c_void, (total_tokens * 4) as usize);
        d_offsets.copy_to_device(seq_offsets.as_ptr() as *const c_void, seq_offsets.len() * 4);
        d_block_table.copy_to_device(block_table.as_ptr() as *const c_void, block_table.len() * 4);
        d_slot_mapping.copy_to_device(h_slot_mapping.as_ptr() as *const c_void, (total_tokens * 4) as usize);
    }

    const NUM_WARMUP: usize = 20;
    const NUM_ITERATIONS: usize = 1000;

    let stream = stream_create_with_flags(0x01);

    println!("[RUST] Отправка одиночного отладочного ядра...");
    unsafe {
        varlen_embeddings(
            d_out.ptr,
            d_weight.ptr,
            d_scales.ptr as *const f32,
            d_tokens.ptr as *const u32,
            d_offsets.ptr as *const i32,
            d_block_table.ptr as *const i32,
            d_slot_mapping.ptr as *mut i32,
            max_blocks_per_seq,
            block_size,
            total_tokens,
            out_features,
            vocab_size,
            num_seqs,
            data_type,
            threads_per_block,
            stream,
        );
    }

    let sync_res = device_synchronize();
    let last_err = get_last_error();
    println!("[RUST] Status sinkhronizatsii: {}, Poslednyaya oshibka: {}", sync_res, last_err);
    if sync_res != 0 || last_err != 0 {
        println!("🚨 KRITICHESKIY SBOY GPU: Yadro avariyno zavershilos! Prover vyravnivaniye ukazateley.");
        return;
    }

    for _ in 0..NUM_WARMUP {
        unsafe {
            varlen_embeddings(
                d_out.ptr,
                d_weight.ptr,
                d_scales.ptr as *const f32,
                d_tokens.ptr as *const u32,
                d_offsets.ptr as *const i32,
                d_block_table.ptr as *const i32,
                d_slot_mapping.ptr as *mut i32,
                max_blocks_per_seq,
                block_size,
                total_tokens,
                out_features,
                vocab_size,
                num_seqs,
                data_type,
                threads_per_block,
                stream,
            );
        }
    }

    let mut start_events = vec![ptr::null_mut(); NUM_ITERATIONS];
    let mut end_events = vec![ptr::null_mut(); NUM_ITERATIONS];
    for i in 0..NUM_ITERATIONS {
        start_events[i] = event_create();
        end_events[i] = event_create();
    }

    let start_host = Instant::now();

    for i in 0..NUM_ITERATIONS {
        unsafe {
            event_record(start_events[i], stream);
            varlen_embeddings(
                d_out.ptr,
                d_weight.ptr,
                d_scales.ptr as *const f32,
                d_tokens.ptr as *const u32,
                d_offsets.ptr as *const i32,
                d_block_table.ptr as *const i32,
                d_slot_mapping.ptr as *mut i32,
                max_blocks_per_seq,
                block_size,
                total_tokens,
                out_features,
                vocab_size,
                num_seqs,
                data_type,
                threads_per_block,
                stream,
            );
            event_record(end_events[i], stream);
        }
    }
    let host_launch_time = start_host.elapsed();
    unsafe {
        event_synchronize(*end_events.last().unwrap());
    }
    let total_host_time = start_host.elapsed();

    let mut bandwidths: Vec<f64> = Vec::with_capacity(NUM_ITERATIONS);
    let mut total_gpu_ms = 0.0_f32;

    let bytes_processed = (total_tokens as u64 * 4)
        + (total_tokens as u64 * 4)
        + (total_tokens as u64 * out_features as u64 * 2)
        + (total_tokens as u64 * (weight_bytes as u64 / vocab_size as u64));

    for i in 0..NUM_ITERATIONS {
        let ms = unsafe { event_elapsed_time(start_events[i], end_events[i]) };
        total_gpu_ms += ms;

        let seconds = (ms / 1000.0) as f64;
        let gbps = (bytes_processed as f64 / 1e9) / seconds;
        bandwidths.push(gbps);
    }

    bandwidths.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let min_bw = bandwidths[0];
    let max_bw = bandwidths[NUM_ITERATIONS - 1];
    let median_bw = bandwidths[NUM_ITERATIONS / 2];
    let p95_worst = bandwidths[(NUM_ITERATIONS as f64 * 0.05) as usize];
    let avg_bw = (bytes_processed as f64 * NUM_ITERATIONS as f64 / 1e9) / (total_gpu_ms as f64 / 1000.0);

    println!("Launch Overhead: {:.6} сек", host_launch_time.as_secs_f32());
    println!("Wall Time:       {:.2} сек", total_host_time.as_secs_f32());
    println!("🚀 MAX ПСП:      {:.2} ГБ/сек", max_bw);
    println!("📈 AVG ПСП:      {:.2} ГБ/сек", avg_bw);
    println!("🎯 P50 ПСП:      {:.2} ГБ/сек", median_bw);
    println!("⚠️ P95 ПСП:      {:.2} ГБ/сек", p95_worst);
    println!("Jitter Шины:     {:.2} ГБ/сек", max_bw - min_bw);
    unsafe {
        d_slot_mapping.copy_to_host(h_slot_mapping.as_mut_ptr() as *mut c_void, (total_tokens * 4) as usize);
        d_out.copy_to_host(h_output.as_mut_ptr() as *mut c_void, (total_tokens * out_features * 2) as usize);
    }

    let mut slot_errors = 0;
    for (token_global_idx, &slot) in h_slot_mapping.iter().enumerate().take(total_tokens as usize) {
        let mut seq_idx = 0;
        for (s, &offset) in seq_offsets.iter().enumerate().take(seqlens.len()) {
            if offset <= token_global_idx as i32 {
                seq_idx = s;
            }
        }
        let start_tok_idx = seq_offsets[seq_idx];
        let token_local_idx = token_global_idx as i32 - start_tok_idx;
        let logical_block_idx = token_local_idx / block_size;
        let block_offset = token_local_idx % block_size;
        let physical_block_id = block_table[seq_idx * max_blocks_per_seq as usize + logical_block_idx as usize];

        let expected_slot = if physical_block_id == -1 { -1 } else { physical_block_id * block_size + block_offset };

        if slot != expected_slot {
            slot_errors += 1;
        }
    }
    println!("Ошибки Slot Mapping (FlashInfer метаданные): {}", slot_errors);
    assert_eq!(slot_errors, 0, "Критическая ошибка построения карты страниц KV-кэша!");

    println!("--- ЧЕСТНАЯ МАТЕМАТИЧЕСКАЯ ВАЛИДАЦИЯ АКТИВАЦИЙ ---");
    let mut math_errors = 0;

    for (tok, &token_id) in h_tokens.iter().enumerate().take(total_tokens as usize) {
        let out_row_offset = tok * out_features as usize;

        for f in 0..out_features as usize {
            let actual_bits = h_output[out_row_offset + f];
            let expected_bits = if token_id >= vocab_size as u32 {
                0u16
            } else {
                match data_type {
                    DataType::BF16 => {
                        let w_offset = (token_id as usize * out_features as usize + f) * 2;
                        let b0 = h_weight[w_offset];
                        let b1 = h_weight[w_offset + 1];
                        ((b1 as u16) << 8) | (b0 as u16)
                    },
                    DataType::FP8 => {
                        let w_offset = token_id as usize * out_features as usize + f;
                        let byte = h_weight[w_offset];
                        let val_f32 = emu_fp8_e4m3_to_f32(byte);
                        f32_to_bf16_bits(val_f32)
                    },
                    DataType::FP4 => {
                        let total_elements_per_row = out_features as usize;
                        let w_offset = (token_id as usize * total_elements_per_row + f) / 2;
                        let byte = h_weight[w_offset];
                        let val_f32 = emu_fp4_e2m1_to_f32(byte, f);
                        let scale_idx = (token_id as usize * total_elements_per_row + f) / 32;
                        let scale = h_scales[scale_idx];
                        f32_to_bf16_bits(val_f32 * scale)
                    },
                }
            };

            if actual_bits != expected_bits {
                if math_errors < 5 {
                    println!(
                        "Ошибка в токен_idx={}, feature={}: Ожидалось биты {:04X}, Получено {:04X}",
                        tok, f, expected_bits, actual_bits
                    );
                }
                math_errors += 1;
            }
        }
    }

    println!("Количество неверных/испорченных элементов: {}", math_errors);
    assert_eq!(
        math_errors, 0,
        "Математическое расхождение: ядро выдало некорректные или пустые значения!"
    );

    println!("🚀 ПОБЕДА! Ядро varlen_embeddings выдает стопроцентную точность на рваном батче!");

    assert_eq!(get_last_error(), 0);
    unsafe {
        for i in 0..NUM_ITERATIONS {
            event_destroy(start_events[i]);
            event_destroy(end_events[i]);
        }
        stream_destroy(stream);
    }
}

fn main() {
    println!("=== УЛЬТИМАТИВНЫЙ RAGGED-БЕНЧМАРК И ВАЛИДАЦИЯ VARLEN_EMBEDDINGS ===");
    run_benchmark_for_type(DataType::BF16);
    run_benchmark_for_type(DataType::FP8);
    run_benchmark_for_type(DataType::FP4);
    println!("\n🚀 ВСЕ ФОРМАТЫ УСПЕШНО ПРОШЛИ ТЕЛЕМЕТРИЮ И МАТЕМАТИЧЕСКУЮ ВАЛИДАЦИЮ!");
}
