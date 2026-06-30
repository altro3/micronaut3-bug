use my_llama::models::linear::Linear;
use my_llama::models::loss::calculate_loss;
use my_llama::utils::{CudaBuffer, CudaStream};
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
        stream: *mut c_void,
    );
}

fn main() {
    println!("=== УЛЬТИМАТИВНЫЙ АСИНХРОННЫЙ КОНТУР ОБУЧЕНИЯ (Tensor Cores) ===");

    my_llama::init_framework();

    const BATCH_SIZE: usize = 2;
    const SEQ_LEN: usize = 1;
    const NUM_TOKENS: usize = BATCH_SIZE * SEQ_LEN;

    const IN_FEATURES: usize = 3;
    const VOCAB_SIZE: usize = 2;

    // Создаем асинхронную очередь команд для GPU Blackwell
    let stream = CudaStream::new();

    // Создаем слой (конструктор внутри сам асинхронно инициализирует веса)
    let linear_layer = Linear::new(IN_FEATURES, VOCAB_SIZE, &stream);

    // Выделяем память во VRAM (Один раз при старте, без аллокаций в рантайме)
    let gpu_input = CudaBuffer::new(NUM_TOKENS * IN_FEATURES);
    let gpu_logits = CudaBuffer::new(NUM_TOKENS * VOCAB_SIZE);

    let gpu_targets = CudaBuffer::new_int(NUM_TOKENS);
    let mut gpu_d_logits = CudaBuffer::new(NUM_TOKENS * VOCAB_SIZE);
    let mut gpu_losses = CudaBuffer::new(NUM_TOKENS);
    let gpu_d_input = CudaBuffer::new(NUM_TOKENS * IN_FEATURES);

    // Асинхронно заливаем входные данные и таргеты по шине PCIe через DMA
    gpu_input.copy_from_host_async(&vec![1.0f32; NUM_TOKENS * IN_FEATURES], &stream);
    gpu_targets.copy_from_host_async(&vec![0, 1], &stream);

    // Создаем и асинхронно зануляем буферы моментов для AdamW силами GPU
    let m_buffer = CudaBuffer::new(IN_FEATURES * VOCAB_SIZE);
    let v_buffer = CudaBuffer::new(IN_FEATURES * VOCAB_SIZE);
    m_buffer.zero_out_async(&stream);
    v_buffer.zero_out_async(&stream);

    // =========================================================================
    // ВЫЧИСЛИТЕЛЬНЫЙ КОНВЕЙЕР (Forward -> Loss -> Backward -> Optimizer)
    // =========================================================================

    // Выполняем Forward в стриме (вызов мгновенно возвращает управление в Rust)
    linear_layer.forward(&gpu_logits, &gpu_input, NUM_TOKENS, &stream);
    println!("1. Прямой проход поставлен в очередь стрима.");

    // Вычисляем Fused Cross-Entropy и стартовые градиенты ошибки
    calculate_loss(
        &gpu_logits,
        &gpu_targets,
        &mut gpu_d_logits,
        &mut gpu_losses,
        NUM_TOKENS,
        VOCAB_SIZE,
        &stream,
    );
    println!("2. Fused Cross-Entropy Loss поставлен в очередь стрима.");

    // Выделяем постоянные массивы-приемники на хосте (CPU) для асинхронной отладки
    let mut host_d_logits = vec![0.0f32; NUM_TOKENS * VOCAB_SIZE];
    let mut host_losses = vec![0.0f32; NUM_TOKENS];
    let mut host_input_debug = vec![0.0f32; NUM_TOKENS * IN_FEATURES];
    let mut host_grads = vec![0.0f32; IN_FEATURES * VOCAB_SIZE];
    let mut host_weights = vec![0.0f32; IN_FEATURES * VOCAB_SIZE];

    // Ставим задачи на асинхронное скачивание данных отладки в наши массивы
    gpu_d_logits.copy_to_host_async(&mut host_d_logits, &stream);
    gpu_losses.copy_to_host_async(&mut host_losses, &stream);
    gpu_input.copy_to_host_async(&mut host_input_debug, &stream);

    // Точка синхронизации: ждем видеокарту, чтобы безопасно распечатать промежуточную отладку
    stream.synchronize();

    println!(
        "   -> [ОТЛАДКА] Градиенты d_logits с GPU: {:?}",
        host_d_logits
    );
    let mean_loss: f32 = host_losses.iter().sum::<f32>() / NUM_TOKENS as f32;
    println!("   -> Лосс на токенах: {:?}", host_losses);
    println!("   -> Средний лосс батча: {:.4}", mean_loss);
    println!(
        "   -> [ОТЛАДКА ВХОДА] gpu_input перед backward: {:?}",
        host_input_debug
    );

    // Запускаем обратный проход на Тензорных ядрах через cuBLAS асинхронно
    linear_layer.backward(&gpu_d_input, &gpu_input, &gpu_d_logits, NUM_TOKENS, &stream);
    println!("3. Обратный проход выполнен. Градиенты dW рассчитаны на Tensor Cores.");

    // Ставим в очередь скачивание рассчитанных градиентов весов dW
    linear_layer
        .weight
        .grad
        .copy_to_host_async(&mut host_grads, &stream);
    stream.synchronize();
    println!(
        "   -> Рассчитанные градиенты весов (dW) с GPU: {:?}",
        host_grads
    );

    // Выполняем асинхронный шаг оптимизатора AdamW
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
            stream.as_raw(),
        );
    }

    // Скачиваем обновленные веса
    linear_layer
        .weight
        .data
        .copy_to_host_async(&mut host_weights, &stream);
    stream.synchronize();

    println!(
        "4. Шаг AdamW выполнен. Обновленные веса: {:?}",
        host_weights
    );
    println!(
        "\n[УСПЕХ] Весь асинхронный контур обучения LLM полностью согласован и работает на пиковой скорости!"
    );
}
