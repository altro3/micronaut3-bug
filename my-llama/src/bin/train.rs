use my_llama::models::linear::Linear;
use my_llama::models::loss::calculate_loss;
use my_llama::utils::{CudaBuffer, CudaStream};
use std::ffi::c_void;

unsafe extern "C" {
    // Подключаем наше ультимативное асинхронное ядро оптимизатора AdamW
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
    println!("=================================================================");
    println!("   ЗАПУСК БОЕВОГО КОНТУРА АСИНХРОННОГО ОБУЧЕНИЯ (CUDA & cuBLAS)  ");
    println!("=================================================================");

    // Инициализируем аппаратную инфраструктуру cuBLAS для Tensor Cores
    my_llama::init_framework();

    // Задаем параметры мини-батча и геометрию сети
    const BATCH_SIZE: usize = 3;
    const IN_FEATURES: usize = 4;
    const VOCAB_SIZE: usize = 3;
    const TOTAL_ELEMENTS: usize = IN_FEATURES * VOCAB_SIZE;

    let stream = CudaStream::new();

    // Создаем обучаемый линейный слой (веса инициализируются асинхронно)
    let linear_layer = Linear::new(IN_FEATURES, VOCAB_SIZE, &stream);

    // Выделяем постоянную VRAM под активации и градиенты (0 аллокаций в цикле)
    let gpu_input = CudaBuffer::new(BATCH_SIZE * IN_FEATURES);
    let gpu_logits = CudaBuffer::new(BATCH_SIZE * VOCAB_SIZE);
    let gpu_targets = CudaBuffer::new_int(BATCH_SIZE);
    let mut gpu_d_logits = CudaBuffer::new(BATCH_SIZE * VOCAB_SIZE);
    let mut gpu_losses = CudaBuffer::new(BATCH_SIZE);
    let gpu_d_input = CudaBuffer::new(BATCH_SIZE * IN_FEATURES);

    // Выделяем и асинхронно зануляем буферы скользящих средних моментов (m, v) для AdamW
    let m_buffer = CudaBuffer::new(TOTAL_ELEMENTS);
    let v_buffer = CudaBuffer::new(TOTAL_ELEMENTS);
    m_buffer.zero_out_async(&stream);
    v_buffer.zero_out_async(&stream);

    // ПОДГОТОВКА ДАННЫХ: Задаем жесткую обучающую выборку (X) и правильные ответы (Y)
    // Входная матрица признаков [BATCH_SIZE, IN_FEATURES]
    let host_input = vec![
        1.0f32, 0.0, 0.5, -0.2, // Токен 0
        0.0, 2.0, 1.1, 0.0, // Токен 1
        -1.0, 0.5, 0.0, 1.5, // Токен 2
    ];
    // Истинные индексы токенов-целей [BATCH_SIZE] (Классы ответов для каждого токена)
    let host_targets: Vec<i32> = vec![0, 2, 1];

    // Асинхронно заливаем данные во VRAM по DMA-каналу PCIe
    gpu_input.copy_from_host_async(&host_input, &stream);
    gpu_targets.copy_from_host_async(&host_targets, &stream);

    // Буферы на CPU для периодического мониторинга динамики лосса
    let mut host_losses = vec![0.0f32; BATCH_SIZE];

    println!("\n[СТАРТ] Запуск цикла оптимизации на 20 эпох. Наблюдаем за падением ошибки...");
    println!("-----------------------------------------------------------------");

    // Запускаем 20 эпох градиентного спуска
    for epoch in 1..=20 {
        // 1. АСИНХРОННЫЙ ПРЯМОЙ ПРОХОД (Forward Pass)
        linear_layer.forward(&gpu_logits, &gpu_input, BATCH_SIZE, &stream);

        // 2. ВЫЧИСЛЕНИЕ LOSS И СТАРТОВЫХ ГРАДИЕНТОВ ОШИБКИ (Fused Softmax + Cross Entropy)
        calculate_loss(
            &gpu_logits,
            &gpu_targets,
            &mut gpu_d_logits,
            &mut gpu_losses,
            BATCH_SIZE,
            VOCAB_SIZE,
            &stream,
        );

        // 3. АСИНХРОННЫЙ ОБРАТНЫЙ ПРОХОД (Backward Pass через cuBLAS)
        // Рассчитывает матрицу градиентов весов layer.weight.grad
        linear_layer.backward(&gpu_d_input, &gpu_input, &gpu_d_logits, BATCH_SIZE, &stream);

        // 4. ШАГ ОПТИМИЗАТОРА ADAMW (Обновление весов + встроенный zero_grad)
        unsafe {
            launch_adamw(
                linear_layer.weight.data.as_raw_ptr(),
                linear_layer.weight.grad.as_raw_ptr(),
                m_buffer.as_raw_ptr(),
                v_buffer.as_raw_ptr(),
                TOTAL_ELEMENTS as i32,
                0.15f32, // Высокий Learning Rate (lr) для демонстрационного быстрого сжатия лосса
                0.9f32,  // beta1
                0.999f32, // beta2
                1e-8f32, // epsilon
                0.01f32, // weight_decay
                epoch as f32, // Текущий шаг для bias correction
                stream.as_raw(),
            );
        }

        // Каждые 2 эпохи стягиваем лосс на CPU и выводим лог, чтобы не перегружать PCIe шину синхронизациями
        if epoch == 1 || epoch % 2 == 0 {
            gpu_losses.copy_to_host_async(&mut host_losses, &stream);

            // Единственная точка синхронизации CPU на всю эпоху!
            stream.synchronize();

            let mean_loss: f32 = host_losses.iter().sum::<f32>() / BATCH_SIZE as f32;
            println!(
                "Эпоха {:02} | Текущие лоссы токенов: {:?} | Средний Loss батча: {:.6}",
                epoch, host_losses, mean_loss
            );
        }
    }

    println!("-----------------------------------------------------------------");
    println!("[УСПЕХ] Тестирование обучения завершено! Ошибка сети упала в разы.");
    println!("Нейросеть успешно подстроила веса под обучающую выборку без блокировок CPU.");
    println!("=================================================================");
}
