use my_llama::utils::CudaBuffer;
use my_llama::models::linear::Linear;
use std::ffi::c_void;

unsafe extern "C" {
    // Импортируем наше ядро оптимизатора AdamW
    fn launch_adamw(
        weights: *mut c_void, gradients: *mut c_void, m_buffer: *mut c_void, v_buffer: *mut c_void,
        size: i32, lr: f32, beta1: f32, beta2: f32, epsilon: f32, weight_decay: f32, step: f32,
    );
}

fn main() {
    println!("=== ИНТЕГРАЦИОННЫЙ ТЕСТ ОБУЧАЕМОГО СЛОЯ LINEAR НА GPU ===");

    // Конфигурация геометрии слоя
    const BATCH_SIZE: usize = 2;   // Батч из 2 токенов
    const IN_FEATURES: usize = 3;  // 3 входных нейрона
    const OUT_FEATURES: usize = 2; // 2 выходных нейрона

    // 1. Инициализируем наш новый изолированный линейный слой
    let linear_layer = Linear::new(IN_FEATURES, OUT_FEATURES);

    // 2. Выделяем буферы для входных данных и ошибок во VRAM
    let gpu_input = CudaBuffer::new(BATCH_SIZE * IN_FEATURES);
    let gpu_output = CudaBuffer::new(BATCH_SIZE * OUT_FEATURES);
    let gpu_d_output = CudaBuffer::new(BATCH_SIZE * OUT_FEATURES); // Градиент ошибки свыше
    let gpu_d_input = CudaBuffer::new(BATCH_SIZE * IN_FEATURES);   // Сюда прилетит градиент для нижнего слоя

    // 3. Заполняем тестовыми данными на хосте и заливаем на GPU
    gpu_input.copy_from_host(&vec![1.0f32; BATCH_SIZE * IN_FEATURES]);
    // Симулируем, что функция потерь выдала ошибку по выходу слоя равенную 0.2
    gpu_d_output.copy_from_host(&vec![0.2f32; BATCH_SIZE * OUT_FEATURES]);

    // Выделяем буферы под моменты AdamW для весов слоя
    let m_buffer = CudaBuffer::new(IN_FEATURES * OUT_FEATURES);
    let v_buffer = CudaBuffer::new(IN_FEATURES * OUT_FEATURES);
    m_buffer.copy_from_host(&vec![0.0f32; IN_FEATURES * OUT_FEATURES]);
    v_buffer.copy_from_host(&vec![0.0f32; IN_FEATURES * OUT_FEATURES]);

    // --- ПРЯМОЙ ПРОХОД (FORWARD) ---
    linear_layer.forward(&gpu_output, &gpu_input, BATCH_SIZE);
    println!("Прямой проход (Forward pass) выполнен на GPU.");

    // --- ОБРАТНЫЙ ПРОХОД (BACKWARD) ---
    // Вычисляем реальные градиенты весов на основе входа и ошибки выхода!
    linear_layer.backward(&gpu_d_input, &gpu_input, &gpu_d_output, BATCH_SIZE);
    println!("Обратный проход (Backward pass) выполнен. Градиенты рассчитаны.");

    // Скачиваем рассчитанные видеокартой градиенты весов для контроля математики
    let calculated_grads = linear_layer.weight.grad.copy_to_host();
    println!("Рассчитанные градиенты весов (dW) с GPU: {:?}", calculated_grads);

    // --- ШАГ ОБНОВЛЕНИЯ ВЕСОВ (OPTIMIZATION) ---
    unsafe {
        launch_adamw(
            linear_layer.weight.data.as_raw_ptr(),
            linear_layer.weight.grad.as_raw_ptr(),
            m_buffer.as_raw_ptr(),
            v_buffer.as_raw_ptr(),
            (IN_FEATURES * OUT_FEATURES) as i32,
            0.01f32, 0.9f32, 0.999f32, 1e-8f32, 0.0f32, 1.0f32,
        );
    }

    let updated_weights = linear_layer.weight.data.copy_to_host();
    println!("Обновленные веса слоя после шага AdamW: {:?}", updated_weights);

    // Верификация математики dW:
    // Каждая ячейка dW = sum по батчу (input * d_output) = 1.0 * 0.2 + 1.0 * 0.2 = 0.4
    if (calculated_grads[0] - 0.4f32).abs() < 1e-5 {
        println!("\n[СКВОЗНОЙ УСПЕХ] Слой Linear успешно прошел цикл Forward -> Backward -> Optimize!");
        println!("Градиенты рассчитаны аппаратно, и веса скорректированы на GPU Blackwell.");
    } else {
        println!("\n[МАТЕМАТИЧЕСКИЙ СБОЙ] Ошибка в расчете транспонированного матричного умножения dW.");
    }
}
