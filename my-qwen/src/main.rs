pub mod utils;
pub mod token;
pub mod models;

use utils::CudaBuffer;
use models::kv_cache::KvCache;

fn main() {
    println!("=== ИНТЕГРАЦИОННЫЙ ТЕСТ KV-CACHE НА GPU ===");

    // Базовые параметры теста
    const MAX_SEQ_LEN: usize = 3;  // Максимальный контекст — 3 токена
    const HIDDEN_SIZE: usize = 2;  // Размерность KV-векторов для одного токена

    // 1. Инициализируем пустой KV-Cache во VRAM
    println!("Выделяем статическую память под KV-Cache на видеокарте...");
    let mut cache = KvCache::new(MAX_SEQ_LEN, HIDDEN_SIZE);
    println!("Начальная позиция контекста: {}", cache.len());

    // 2. Симулируем появление ПЕРВОГО токена.
    // Пусть его вектор Ключа будет [1.0, 1.0], а Значения — [1.1, 1.1]
    let k1 = vec![1.0f32, 1.0f32];
    let v1 = vec![1.1f32, 1.1f32];

    let gpu_k1 = CudaBuffer::new(HIDDEN_SIZE);
    let gpu_v1 = CudaBuffer::new(HIDDEN_SIZE);
    gpu_k1.copy_from_host(&k1);
    gpu_v1.copy_from_host(&v1);

    println!("\n[Токен 1] Добавляем новые векторы в кэш...");
    cache.append(&gpu_k1, &gpu_v1);
    println!("Текущая длина контекста: {}", cache.len());

    // 3. Симулируем появление ВТОРОГО токена.
    // Его векторы: Ключ = [2.0, 2.0], Значение = [2.2, 2.2]
    let k2 = vec![2.0f32, 2.0f32];
    let v2 = vec![2.2f32, 2.2f32];

    let gpu_k2 = CudaBuffer::new(HIDDEN_SIZE);
    let gpu_v2 = CudaBuffer::new(HIDDEN_SIZE);
    gpu_k2.copy_from_host(&k2);
    gpu_v2.copy_from_host(&v2);

    println!("\n[Токен 2] Добавляем новые векторы в кэш...");
    cache.append(&gpu_k2, &gpu_v2);
    println!("Текущая длина контекста: {}", cache.len());

    // 4. Скачиваем ВЕСЬ кэш целиком из видеокарты для верификации смещений памяти
    let full_k_cache = cache.k_cache.copy_to_host();
    let full_v_cache = cache.v_cache.copy_to_host();

    println!("\nФинальный слепок K-Cache с GPU: {:?}", full_k_cache);
    println!("Финальный слепок V-Cache с GPU: {:?}", full_v_cache);

    // Ожидаемый результат в памяти:
    // Токен 1 лег в индекс 0..2, Токен 2 лег в индекс 2..4, а индекс 4..6 остался нулевым (пустым)
    let expected_k = vec![1.0f32, 1.0f32, 2.0f32, 2.0f32, 0.0f32, 0.0f32];
    let expected_v = vec![1.1f32, 1.1f32, 2.2f32, 2.2f32, 0.0f32, 0.0f32];

    if full_k_cache == expected_k && full_v_cache == expected_v {
        println!("\n[УСПЕХ] KV-Cache идеально распределил векторы по смещениям на GPU!");
        println!("Контекст защищен от перезаписи, память выделена без фрагментации.");
    } else {
        println!("\n[КРИТИЧЕСКАЯ ОШИБКА] Данные токенов наложились друг на друга или смешались.");
    }
}
