use my_llama::tokenizer::factory::TokenizerFactory;
use my_llama::tokenizer::trainer::{BpeTrainer, TrainerConfig};
use std::fs;
use std::time::Instant;

fn main() -> std::io::Result<()> {
    let input_path = "data/input.txt";
    let model_path = "data/my_bpe_model.json";

    fs::create_dir_all("data")?;

    if !std::path::Path::new(input_path).exists() {
        println!("[Tokenizer Test] Создаю демонстрационный файл текста...");
        fs::write(
            input_path,
            "Привет мир! Это тестовый большой реальный текст для проверки нашего токенизатора на Rust. ".repeat(30000),
        )?;
    }

    let text_content = fs::read_to_string(input_path)?;
    let total_bytes = text_content.len();
    println!(
        "[Tokenizer Test] Размер исходного текста: {:.2} МБ",
        total_bytes as f64 / (1024.0 * 1024.0)
    );

    // 1. Обучаем словарь с использованием TrainerConfig уровня 2026 года
    let config = TrainerConfig::default();
    let trainer = BpeTrainer::new(200000, config);

    let start_train = Instant::now();
    trainer.train(&text_content, model_path)?;
    println!("[Tokenizer Test] Время обучения словаря: {:?}", start_train.elapsed());

    // 2. Загружаем свежесозданный словарь через нашу AVX2 фабрику
    println!("\n[Tokenizer Test] Загружаем свежесозданный словарь через фабрику...");
    let tokenizer = TokenizerFactory::from_file(model_path)?;

    // --- СУПЕРСКАЛЯРНЫЙ МНОГОПОТОЧНЫЙ ТЕСТ ---
    println!("\n[Tokenizer Test] Эмулируем батчинг: нарезаем текст на 64 независимые строки...");
    let mut batch_texts = Vec::with_capacity(64);
    let chunk_size = text_content.len() / 64;
    let mut current_idx = 0;

    for _ in 0..63 {
        let mut end_idx = current_idx + chunk_size;
        while end_idx < text_content.len() && !text_content.is_char_boundary(end_idx) {
            end_idx += 1;
        }
        batch_texts.push(text_content[current_idx..end_idx].to_string());
        current_idx = end_idx;
    }
    // Хвост забираем целиком без брейк-поинтов
    if current_idx < text_content.len() {
        batch_texts.push(text_content[current_idx..].to_string());
    }

    let batch_bytes: usize = batch_texts.iter().map(|s| s.len()).sum();
    println!("[Tokenizer Test] Токенизируем БАТЧ в МНОГОПОТОЧНОМ режиме...");
    println!("  |-> Всего строк на обработку: {}", batch_texts.len());
    println!(
        "  |-> Средний размер одной строки: {:.2} МБ",
        (batch_bytes as f64 / batch_texts.len() as f64) / (1024.0 * 1024.0)
    );

    // Запуск нашего полностью изолированного, параллельного энкодера
    let start_parallel = Instant::now();
    let parallel_results = tokenizer.encode_parallel(&batch_texts);
    let duration_parallel = start_parallel.elapsed();

    let total_tokens: usize = parallel_results.iter().map(|v| v.len()).sum();
    let speed_parallel = (batch_bytes as f64 / 1024.0 / 1024.0) / duration_parallel.as_secs_f64();

    println!("\n|-> СИНХРОНИЗАЦИЯ ПОТОКОВ ЗАВЕРШЕНА");
    println!("|-> Суммарно получено токенов: {}", total_tokens);
    println!("|-> Время параллельного кодирования: {:?}", duration_parallel);
    println!(
        "|-> Скорость параллельного кодирования: {:.2} MiB/s ({:.2} GiB/s)",
        speed_parallel,
        speed_parallel / 1024.0
    );

    Ok(())
}
