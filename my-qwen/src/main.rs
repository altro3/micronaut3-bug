pub mod utils;
pub mod token;
pub mod models;

use utils::CudaBuffer;
use models::RmsNorm;

fn main() {
    println!("=== ИНТЕГРАЦИОННЫЙ ТЕСТ СЛОЯ RMSNorm НА GPU ===");

    // Параметры нашей микро-модели
    const BATCH_SIZE: usize = 2;   // Сымитируем контекст из 2 токенов
    const HIDDEN_SIZE: usize = 4;  // Размерность векторов скрытого состояния слоя

    // 1. Создаем тестовую матрицу на хосте (2 строки по 4 элемента)
    // Специально берем числа с большим разбросом, чтобы увидеть эффект нормализации
    let host_input = vec![
        1.0f32,  2.0f32,  3.0f32,  4.0f32, // Первый токен
        10.0f32, 0.0f32, -5.0f32,  2.0f32, // Второй токен
    ];
    println!("Исходная матрица (Host RAM):\n  Токен 1: {:?}\n  Токен 2: {:?}",
             &host_input[0..4], &host_input[4..8]);

    // 2. Выделяем память во VRAM под входные и выходные данные
    let total_elements = BATCH_SIZE * HIDDEN_SIZE;
    let gpu_input = CudaBuffer::new(total_elements);
    let gpu_output = CudaBuffer::new(total_elements);

    // Копируем исходные данные во VRAM видеокарты
    gpu_input.copy_from_host(&host_input);

    // 3. Инициализируем наш промышленный слой нормализации
    println!("\nИнициализируем слой RmsNorm во VRAM...");
    let rms_norm_layer = RmsNorm::new(HIDDEN_SIZE);

    // 4. Запускаем математический расчет прямо на чипе GPU
    println!("Выполняем прямой проход (forward) на видеокарте...");
    rms_norm_layer.forward(&gpu_output, &gpu_input, BATCH_SIZE);

    // 5. Скачиваем результат вычислений обратно в RAM
    let host_output = gpu_output.copy_to_host();
    println!("\nРезультат нормализации с GPU:");
    println!("  Токен 1: {:?}", &host_output[0..4]);
    println!("  Токен 2: {:?}", &host_output[4..8]);

    // 6. Математическая проверка точности (Unit-тест)
    // Для первого токена:
    // Сумма квадратов = 1+4+9+16 = 30. Ср. квадрат = 30 / 4 = 7.5
    // RMS = sqrt(7.5) = 2.7386127.
    // Первый элемент должен быть примерно: 1.0 / 2.7386127 * 1.0 (вес) = 0.365148
    let expected_t1_e1 = 1.0f32 / (30.0f32 / 4.0f32).sqrt();
    let actual_t1_e1 = host_output[0];

    println!("\nСверка точности для первого элемента:");
    println!("  Ожидаемое значение:  {:.6}", expected_t1_e1);
    println!("  Фактическое с GPU:   {:.6}", actual_t1_e1);

    // Разница между f32 числами из-за аппаратного округления GPU не должна превышать дельту epsilon
    if (actual_t1_e1 - expected_t1_e1).abs() < 1e-5 {
        println!("\n[МАТЕМАТИЧЕСКИЙ УСПЕХ] Видеокарта выполнила расчет RMSNorm с идеальной точностью!");
    } else {
        println!("\n[КРИТИЧЕСКАЯ ОШИБКА] Математический сбой или искажение данных в регистрах GPU.");
    }
}
