use my_llama::utils::CudaBuffer;
use std::ffi::c_void;

unsafe extern "C" {
    // Импортируем наше параллельное CUDA-ядро AdamW
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
    println!("=== ЗАПУСК ДВИЖКА ОБУЧЕНИЯ (TRAINING LOOP) MY-LLAMA ===");

    // Симулируем один слой из 4 весов
    const PARAM_SIZE: usize = 4;

    // 1. Аллокация памяти во VRAM для обучения
    let weights = CudaBuffer::new(PARAM_SIZE);
    let gradients = CudaBuffer::new(PARAM_SIZE);
    let m_buffer = CudaBuffer::new(PARAM_SIZE); // Первый момент (Momentum)
    let v_buffer = CudaBuffer::new(PARAM_SIZE); // Второй момент (RMSProp)

    // 2. Инициализируем веса и градиенты тестовыми значениями
    weights.copy_from_host(&vec![1.0f32, 1.0f32, 1.0f32, 1.0f32]);
    // Симулируем, что обратный проход (Backward Pass) насчитал ошибку (градиент = 0.1)
    gradients.copy_from_host(&vec![0.1f32, 0.1f32, 0.1f32, 0.1f32]);
    // Буферы моментов изначально заполнены нулями
    m_buffer.copy_from_host(&vec![0.0f32; PARAM_SIZE]);
    v_buffer.copy_from_host(&vec![0.0f32; PARAM_SIZE]);

    println!("Стартовые веса на GPU: [1.0, 1.0, 1.0, 1.0]");
    println!("Насчитанные градиенты: [0.1, 0.1, 0.1, 0.1]");

    // 3. Настройки гиперпараметров обучения AdamW
    let lr = 0.01f32;
    let beta1 = 0.9f32;
    let beta2 = 0.999f32;
    let epsilon = 1e-8f32;
    let weight_decay = 0.01f32;
    let step = 1.0f32; // Первый шаг оптимизации

    println!("\n[GPU] Выполняем шаг оптимизатора AdamW...");
    unsafe {
        launch_adamw(
            weights.as_raw_ptr(),
            gradients.as_raw_ptr(),
            m_buffer.as_raw_ptr(),
            v_buffer.as_raw_ptr(),
            PARAM_SIZE as i32,
            lr,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            step,
        );
    }

    // 4. Скачиваем обновленные веса обратно, чтобы проверить математику шага
    let updated_weights = weights.copy_to_host();
    let updated_gradients = gradients.copy_to_host();

    println!("\nРезультаты обучения с GPU:");
    println!("  Обновленные веса:        Rhine: {:?}", updated_weights);
    println!(
        "  Очищенные градиенты (должны быть 0): {:?}",
        updated_gradients
    );

    // Математический расчет эталона на CPU:
    // g = 0.1, w = 1.0
    // m = 0.9 * 0 + 0.1 * 0.1 = 0.01 -> m_hat = 0.01 / (1 - 0.9^1) = 0.1
    // v = 0.999 * 0 + 0.001 * 0.01 = 0.00001 -> v_hat = 0.00001 / (1 - 0.999^1) = 0.01
    // w = 1.0 - 0.01 * (0.1 / (sqrt(0.01) + 1e-8) + 0.01 * 1.0) = 1.0 - 0.01 * (1.0 + 0.01) = 0.9899
    let expected_weight = 0.9899f32;

    if (updated_weights[0] - expected_weight).abs() < 1e-4 && updated_gradients[0] == 0.0f32 {
        println!(
            "\n[ОБУЧЕНИЕ УСПЕШНО] Кернел AdamW отработал с идеальной точностью, обновил веса и занулил градиенты!"
        );
    } else {
        println!(
            "\n[КРИТИЧЕСКАЯ ОШИБКА] Ошибка в математике обновления моментов или затухания весов."
        );
    }
}
