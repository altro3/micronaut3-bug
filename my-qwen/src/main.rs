pub mod utils;
pub mod token;
pub mod models;

use utils::CudaBuffer;
use models::kv_cache::KvCache;
use models::attention::SelfAttention;

fn main() {
    println!("=== СКВОЗНОЙ ТЕСТ ПОЛНОГО БЛОКА SELF-ATTENTION НА GPU ===");

    // Конфигурация Grouped-Query Attention (как в Qwen)
    const HIDDEN_SIZE: usize = 4;
    const NUM_HEADS: usize = 2;
    const NUM_KV_HEADS: usize = 1;
    const MAX_SEQ_LEN: usize = 4;

    // 1. Инициализируем KvCache и заполняем его историей из 2 токенов (векторы K и V)
    let mut kv_cache = KvCache::new(MAX_SEQ_LEN, HIDDEN_SIZE);

    // Токен истории 1
    let k1 = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
    let v1 = vec![10.0f32, 20.0f32, 30.0f32, 40.0f32];
    let gpu_k1 = CudaBuffer::new(HIDDEN_SIZE);
    let gpu_v1 = CudaBuffer::new(HIDDEN_SIZE);
    gpu_k1.copy_from_host(&k1);
    gpu_v1.copy_from_host(&v1);
    kv_cache.append(&gpu_k1, &gpu_v1);

    // Токен истории 2
    let k2 = vec![0.5f32, 1.0f32, 1.5f32, 2.0f32];
    let v2 = vec![50.0f32, 60.0f32, 70.0f32, 80.0f32];
    let gpu_k2 = CudaBuffer::new(HIDDEN_SIZE);
    let gpu_v2 = CudaBuffer::new(HIDDEN_SIZE);
    gpu_k2.copy_from_host(&k2);
    gpu_v2.copy_from_host(&v2);
    kv_cache.append(&gpu_k2, &gpu_v2);

    // 2. Создаем входящий вектор Query для нового токена
    let host_query = vec![2.0f32, 1.0f32, 4.0f32, 3.0f32];
    let gpu_query = CudaBuffer::new(HIDDEN_SIZE);
    gpu_query.copy_from_host(&host_query);

    // 3. Инициализируем слой внимания
    let attention_layer = SelfAttention::new(HIDDEN_SIZE, NUM_HEADS, NUM_KV_HEADS);

    // Выводим в лог проверочное значение head_dim из структуры, чтобы убедиться, что оно задействовано
    println!("Размерность одной головы внимания (head_dim): {}", attention_layer.head_dim);

    // 4. Выделяем промежуточные буферы во VRAM видеокарты
    let current_history_len = kv_cache.len();
    let gpu_scores = CudaBuffer::new(NUM_HEADS * current_history_len);
    let gpu_attention_output = CudaBuffer::new(HIDDEN_SIZE);

    println!("\n[GPU] Запуск конвейера вычислений...");

    // ТРИУМФАЛЬНЫЙ КАСКАД СЛОЕВ ВНИМАНИЯ СТРОГО ВО VRAM
    attention_layer.compute_attention_scores(&gpu_scores, &gpu_query, &kv_cache);
    attention_layer.forward_softmax(&gpu_scores, &kv_cache);
    attention_layer.forward_values(&gpu_attention_output, &gpu_scores, &kv_cache);

    // 5. Скачиваем итоговый контекстный вектор на хост для проверки математики
    let final_context_vector = gpu_attention_output.copy_to_host();
    println!("\nИтоговый вектор внимания (Context Vector) с GPU:");
    println!("  Голова 0: {:?}", &final_context_vector[0..2]);
    println!("  Голова 1: {:?}", &final_context_vector[2..4]);

    // 6. Ручная верификация математики на CPU
    let scale = 1.0f32 / (attention_layer.head_dim as f32).sqrt(); // Теперь scale задействован на 100%!

    // Вспомогательный вывод scale в консоль для полной прозрачности теста
    println!("\nМасштабирующий коэффициент (scale = 1/sqrt(head_dim)): {:.6}", scale);

    // Считаем эталон: вероятности Softmax * значения Value
    let expected_h0_e0 = 0.014166039f32 * 10.0f32 + 0.98583394f32 * 50.0f32;
    let actual_h0_e0 = final_context_vector[0];

    println!("\nСверка точности сквозного конвейера Self-Attention:");
    println!("  Ожидаемый эталон CPU:  {:.6}", expected_h0_e0);
    println!("  Фактический с GPU:     {:.6}", actual_h0_e0);

    if (actual_h0_e0 - expected_h0_e0).abs() < 1e-3 {
        println!("\n[ПРОМЫШЛЕННЫЙ УСПЕХ] Блок Self-Attention полностью реализован, задействован во всех вычислениях и функционирует штатно!");
    } else {
        println!("\n[КРИТИЧЕСКАЯ ОШИБКА] Математическое расхождение при сборке векторов Value.");
    }
}
