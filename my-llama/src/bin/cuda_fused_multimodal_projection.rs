use std::ffi::c_void;
use std::ptr;
use std::time::Instant;

use cuda_runtime::data_types::DataType;
use cuda_runtime::input_stage::fused_multimodal_projection;
use cuda_runtime::{
    CudaBuffer, device_synchronize, event_create, event_destroy, event_elapsed_time, event_record, event_synchronize, get_last_error,
    stream_create_with_flags, stream_destroy,
};
use my_llama::test_utils::{bf16_bits_to_f32, emu_fp4_e2m1_to_f32, emu_fp8_e4m3_to_f32, f32_to_bf16_bits};

fn run_projection_test(data_type: DataType) {
    let type_name = format!("{:?}", data_type);
    println!("\n=== PROJECTION TELEMETRY FORMAT: {} ===", type_name);

    let vision_hidden_size = 1152;
    let text_hidden_size = 4096;
    let tp_size = 2;
    let tp_rank = 0;
    let num_segments = 3;

    let local_output_dim = text_hidden_size / tp_size;
    let rank_offset = tp_rank * local_output_dim;

    let segment_tokens = vec![576, 1152, 288];

    let mut host_problem_shapes: Vec<i32> = Vec::with_capacity(num_segments * 3);
    let mut total_tokens = 0;
    for &m in &segment_tokens {
        host_problem_shapes.push(m);
        host_problem_shapes.push(local_output_dim);
        host_problem_shapes.push(vision_hidden_size);
        total_tokens += m as usize;
    }

    let mut h_inputs = Vec::new();
    let mut h_weights = Vec::new();
    let mut h_outputs = Vec::new();

    let mut d_inputs = Vec::new();
    let mut d_weights = Vec::new();
    let mut d_outputs = Vec::new();

    let mut algorithmic_bytes: u64 = 0;

    for i in 0..num_segments {
        let m = segment_tokens[i] as usize;
        let k = vision_hidden_size as usize;
        let n = local_output_dim as usize;

        let input_elements = m * k;
        let mut input_segment = vec![0u16; input_elements];
        for (j, item) in input_segment.iter_mut().enumerate() {
            *item = f32_to_bf16_bits(0.1f32 * (j % 7) as f32);
        }

        let weight_bytes = match data_type {
            DataType::BF16 => n * k * 2,
            DataType::FP8 => n * k,
            DataType::FP4 => (n * k) / 2,
        };
        let mut weight_segment = vec![0x2Bu8; weight_bytes];
        if data_type == DataType::BF16 {
            let weight_bf16 = unsafe { std::slice::from_raw_parts_mut(weight_segment.as_mut_ptr() as *mut u16, n * k) };
            for (j, item) in weight_bf16.iter_mut().enumerate() {
                *item = f32_to_bf16_bits(0.02f32 * (j % 5) as f32);
            }
        }

        let output_elements = m * n;
        let output_segment = vec![0u16; output_elements];

        let d_input = CudaBuffer::alloc(input_elements * 2);
        let d_weight = CudaBuffer::alloc(weight_bytes);
        let d_output = CudaBuffer::alloc(output_elements * 2);

        unsafe {
            d_input.copy_to_device(input_segment.as_ptr() as *const c_void, input_elements * 2);
            d_weight.copy_to_device(weight_segment.as_ptr() as *const c_void, weight_bytes);
        }

        algorithmic_bytes += (input_elements * 2) as u64 + weight_bytes as u64 + (output_elements * 2) as u64;

        d_inputs.push(d_input);
        d_weights.push(d_weight);
        d_outputs.push(d_output);

        h_inputs.push(input_segment);
        h_weights.push(weight_segment);
        h_outputs.push(output_segment);
    }

    let mut h_bias = vec![0.0f32; text_hidden_size as usize];
    for (i, item) in h_bias.iter_mut().enumerate() {
        *item = 0.05f32 * (i % 3) as f32;
    }
    let d_bias = CudaBuffer::alloc((text_hidden_size * 4) as usize);
    unsafe {
        d_bias.copy_to_device(h_bias.as_ptr() as *const c_void, (text_hidden_size * 4) as usize);
    }
    algorithmic_bytes += (total_tokens as u64) * (local_output_dim as u64) * 4;

    let h_scales = vec![0.85f32; num_segments];
    let d_scales = CudaBuffer::alloc(num_segments * 4);
    unsafe {
        d_scales.copy_to_device(h_scales.as_ptr() as *const c_void, num_segments * 4);
    }

    let host_ptr_a: Vec<*mut c_void> = d_inputs.iter().map(|b| b.ptr).collect();
    let host_ptr_b: Vec<*mut c_void> = d_weights.iter().map(|b| b.ptr).collect();
    let host_ptr_d: Vec<*mut c_void> = d_outputs.iter().map(|b| b.ptr).collect();

    let raw_ptr_a = host_ptr_a.as_ptr();
    let raw_ptr_b = host_ptr_b.as_ptr();
    let raw_ptr_d = host_ptr_d.as_ptr();

    let raw_table_size = num_segments * 8;
    let table_size = (raw_table_size + 15) & !15;
    let shapes_size = num_segments * 3 * std::mem::size_of::<i32>();
    let required_workspace_bytes = 3 * table_size + shapes_size + 64;

    let d_workspace = CudaBuffer::alloc(required_workspace_bytes);
    let stream = stream_create_with_flags(0x01);

    assert_eq!(device_synchronize(), 0);
    assert_eq!(get_last_error(), 0);

    const WARMUP: usize = 20;
    const ITERS: usize = 200;

    for _ in 0..WARMUP {
        unsafe {
            fused_multimodal_projection(
                raw_ptr_a,
                raw_ptr_b,
                raw_ptr_d,
                d_bias.ptr as *const f32,
                d_scales.ptr as *const f32,
                host_problem_shapes.as_ptr(),
                num_segments as i32,
                vision_hidden_size,
                text_hidden_size,
                tp_rank,
                tp_size,
                data_type,
                d_workspace.ptr,
                stream,
            );
        }
    }
    assert_eq!(device_synchronize(), 0);

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
            fused_multimodal_projection(
                raw_ptr_a,
                raw_ptr_b,
                raw_ptr_d,
                d_bias.ptr as *const f32,
                d_scales.ptr as *const f32,
                host_problem_shapes.as_ptr(),
                num_segments as i32,
                vision_hidden_size,
                text_hidden_size,
                tp_rank,
                tp_size,
                data_type,
                d_workspace.ptr,
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
        let ms = unsafe { event_elapsed_time(start_events[i], end_events[i]) };
        total_gpu_ms += ms;
    }

    let avg_gpu_ms = total_gpu_ms / (ITERS as f32);
    let avg_host_ms = (total_host_time.as_secs_f32() * 1000.0f32) / (ITERS as f32);
    let tflops = if avg_gpu_ms > 0.0 {
        let mut total_ops = 0.0f64;
        for i in 0..num_segments {
            total_ops += 2.0 * (segment_tokens[i] as f64) * (local_output_dim as f64) * (vision_hidden_size as f64);
        }
        (total_ops * 1e-12) / (avg_gpu_ms as f64 * 1e-3)
    } else {
        0.0
    };

    let bandwidth_gbps = if avg_gpu_ms > 0.0 {
        (algorithmic_bytes as f64 * 1e-9) / (avg_gpu_ms as f64 * 1e-3)
    } else {
        0.0
    };

    println!("  - Avg GPU Execution Time: {:.3} ms", avg_gpu_ms);
    println!("  - Avg Host Dispatch Time:  {:.3} ms", avg_host_ms);
    println!("  - Algorithmic Performance: {:.2} TFLOPS", tflops);
    println!("  - Useful Memory Bandwidth:  {:.2} GB/s", bandwidth_gbps);
    let mut mut_h_outputs = h_outputs;
    for i in 0..num_segments {
        let output_elements = (segment_tokens[i] as usize) * (local_output_dim as usize);
        unsafe {
            d_outputs[i].copy_to_host(mut_h_outputs[i].as_mut_ptr() as *mut c_void, output_elements * 2);
        }
    }

    println!("--- MULTIMODAL PROJECTION MATHEMATICAL VALIDATION ---");
    let mut math_errors = 0;
    let allowed_tolerance = 5e-2f32;

    for s in 0..num_segments {
        let m = segment_tokens[s] as usize;
        let k = vision_hidden_size as usize;
        let n = local_output_dim as usize;
        let scale = h_scales[s];

        for row in 0..m {
            for col in 0..n {
                let mut accum = 0.0f32;

                for contr in 0..k {
                    let input_val = bf16_bits_to_f32(h_inputs[s][row * k + contr]);

                    let weight_val = match data_type {
                        DataType::BF16 => {
                            let weight_bf16 = unsafe { std::slice::from_raw_parts(h_weights[s].as_ptr() as *const u16, n * k) };
                            bf16_bits_to_f32(weight_bf16[col * k + contr])
                        },
                        DataType::FP8 => {
                            let raw_fp8 = h_weights[s][col * k + contr];
                            let dequantized_weight = emu_fp8_e4m3_to_f32(raw_fp8) * scale;
                            let bf16_bits = f32_to_bf16_bits(dequantized_weight);
                            bf16_bits_to_f32(bf16_bits)
                        },
                        DataType::FP4 => {
                            let linear_idx = col * k + contr;
                            let byte_idx = linear_idx / 2;
                            let packed_byte = h_weights[s][byte_idx];
                            let sub_byte_offset = linear_idx % 2;
                            let raw_fp4 = (packed_byte >> (sub_byte_offset * 4)) & 0x0F;
                            emu_fp4_e2m1_to_f32(raw_fp4, linear_idx)
                        },
                    };

                    accum += input_val * weight_val;
                }

                if data_type == DataType::FP4 {
                    accum *= scale;
                }

                let bias_val = h_bias[rank_offset as usize + col];
                let val_with_bias = accum + bias_val;
                let expected_raw = val_with_bias / (1.0f32 + (-val_with_bias).exp());

                let expected_bf16_bits = f32_to_bf16_bits(expected_raw);
                let expected = bf16_bits_to_f32(expected_bf16_bits);

                let actual = bf16_bits_to_f32(mut_h_outputs[s][row * n + col]);
                let diff = (actual - expected).abs();

                if diff > allowed_tolerance {
                    if math_errors < 5 {
                        println!(
                            "Mismatch at seg {}, row {}, col {}: GPU={}, CPU={}, Diff={}",
                            s, row, col, actual, expected, diff
                        );
                    }
                    math_errors += 1;
                }
            }
        }
    }
    println!("Projection Mismatch Count: {}", math_errors);
    assert_eq!(math_errors, 0, "Critical math mismatch in Multimodal Projection!");
    println!("✅ MULTIMODAL PROJECTION VALIDATION PASSED FOR {}", type_name);

    unsafe {
        for i in 0..ITERS {
            event_destroy(start_events[i]);
            event_destroy(end_events[i]);
        }
        stream_destroy(stream);
    }
}

pub fn main() {
    run_projection_test(DataType::BF16);
    run_projection_test(DataType::FP8);
    run_projection_test(DataType::FP4);
}
