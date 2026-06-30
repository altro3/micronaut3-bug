use my_llama::models::linear::Linear;
use my_llama::models::loss::calculate_loss;
use my_llama::utils::CudaBuffer;
use std::ffi::c_void;

unsafe extern "C" {
    fn launch_adamw(
        weights: *mut c_void,
        gradients: *mut c_void,
        m_buffer: *mut c_void,
        v_buffer: *mut c_void,
        size: i32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
        step: f32,
    );
}

fn main() {
    println!("=== ИНТЕГРАЦИОННЫЙ ТЕСТ ОБУЧЕНИЯ С FUSED CROSS-ENTROPY LOSS ===");

    const BATCH_SIZE: usize = 2;
    const SEQ_LEN: usize = 1;
    const NUM_TOKENS: usize = BATCH_SIZE * SEQ_LEN;

    const IN_FEATURES: usize = 3;
    const VOCAB_SIZE: usize = 2;

    let linear_layer = Linear::new(IN_FEATURES, VOCAB_SIZE);

    let gpu_input = CudaBuffer::new(NUM_TOKENS * IN_FEATURES);
    let gpu_logits = CudaBuffer::new(NUM_TOKENS * VOCAB_SIZE);

    let gpu_targets = CudaBuffer::new_int(NUM_TOKENS);
    let mut gpu_d_logits = CudaBuffer::new(NUM_TOKENS * VOCAB_SIZE);
    let mut gpu_losses = CudaBuffer::new(NUM_TOKENS);

    let gpu_d_input = CudaBuffer::new(NUM_TOKENS * IN_FEATURES);

    gpu_input.copy_from_host(&vec![1.0f32; NUM_TOKENS * IN_FEATURES]);

    let host_targets: Vec<i32> = vec![0, 1];
    gpu_targets.copy_from_host_int(&host_targets);

    let m_buffer = CudaBuffer::new(IN_FEATURES * VOCAB_SIZE);
    let v_buffer = CudaBuffer::new(IN_FEATURES * VOCAB_SIZE);
    m_buffer.copy_from_host(&vec![0.0f32; IN_FEATURES * VOCAB_SIZE]);
    v_buffer.copy_from_host(&vec![0.0f32; IN_FEATURES * VOCAB_SIZE]);

    linear_layer.forward(&gpu_logits, &gpu_input, NUM_TOKENS);
    println!("1. Прямой проход выполнен. Логиты посчитаны на GPU.");

    calculate_loss(
        &gpu_logits,
        &gpu_targets,
        &mut gpu_d_logits,
        &mut gpu_losses,
        NUM_TOKENS,
        VOCAB_SIZE,
    );
    println!("2. Fused Cross-Entropy Loss успешно выполнен.");

    let debug_d_logits = gpu_d_logits.copy_to_host();
    println!("   -> [ОТЛАДКА] Градиенты d_logits с GPU: {:?}", debug_d_logits);

    let losses_host = gpu_losses.copy_to_host();
    let mean_loss: f32 = losses_host.iter().sum::<f32>() / NUM_TOKENS as f32;
    println!("   -> Лосс на токенах: {:?}", losses_host);
    println!("   -> Средний лосс батча: {:.4}", mean_loss);

    linear_layer.backward(&gpu_d_input, &gpu_input, &gpu_d_logits, NUM_TOKENS);
    println!("3. Обратный проход выполнен. Градиенты весов dW рассчитаны через p_i - y_i.");

    let calculated_grads = linear_layer.weight.grad.copy_to_host();
    println!(
        "   -> Рассчитанные градиенты весов (dW) с GPU: {:?}",
        calculated_grads
    );

    unsafe {
        launch_adamw(
            linear_layer.weight.data.as_raw_ptr(),
            linear_layer.weight.grad.as_raw_ptr(),
            m_buffer.as_raw_ptr(),
            v_buffer.as_raw_ptr(),
            (IN_FEATURES * VOCAB_SIZE) as i32,
            0.01f32,
            0.9f32,
            0.999f32,
            1e-8f32,
            0.0f32,
            1.0f32,
        );
    }

    let updated_weights = linear_layer.weight.data.copy_to_host();
    println!(
        "4. Шаг AdamW выполнен. Обновленные веса: {:?}",
        updated_weights
    );
    println!("\n[УСПЕХ] Интеграция Лосса и Линейного слоя работает в едином контуре обучения!");
}
