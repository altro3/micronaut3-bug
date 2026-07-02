use my_llama::tokenizer::factory::TokenizerFactory;
use my_llama::tokenizer::BpeTrainer;
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
            "Привет мир! Это тестовый большой реальный текст для проверки нашего токенизатора на Rust. ".repeat(15000),
        )?;
    }

    let text_content = fs::read_to_string(input_path)?;
    let total_bytes = text_content.len();
    println!(
        "[Tokenizer Test] Размер исходного текста: {:.2} МБ",
        total_bytes as f64 / (1024.0 * 1024.0)
    );

    // 1. Обучаем словарь
    let trainer = BpeTrainer::new(200000);
    let start_train = Instant::now();
    trainer.train(&text_content, model_path)?;
    println!("[Tokenizer Test] Время обучения словаря: {:?}", start_train.elapsed());

    // 2. Загружаем свежесозданный словарь через фабрику
    println!("\n[Tokenizer Test] Загружаем свежесозданный словарь через фабрику...");
    let tokenizer = TokenizerFactory::from_file(model_path)?;

    // --- ОДНОПОТОЧНЫЙ ТЕСТ ---
    // println!("[Tokenizer Test] Токенизируем текст в ОДИН поток...");
    // let start_encode = Instant::now();
    // let tokens = tokenizer.encode(&text_content);
    // let duration = start_encode.elapsed();
    //
    // let speed = (total_bytes as f64 / 1024.0 / 1024.0) / duration.as_secs_f64();
    // println!("|-> Получено токенов: {}", tokens.len());
    // println!("|-> Время кодирования: {:?}", duration);
    // println!("|-> Скорость: {:.2} MiB/s", speed);
    // println!("|-> Первые 20 токенов (без нулей): {:?}", &tokens[..20.min(tokens.len())]);

    // --- МНОГОПОТОЧНЫЙ ТЕСТ ---
    println!("\n[Tokenizer Test] Эмулируем батчинг: нарезаем текст на 64 независимые строки...");
    let chunk_size = (text_content.len() / 64).max(1);
    let mut batch_texts = Vec::with_capacity(64);
    let mut current_idx = 0;

    while current_idx < text_content.len() {
        let end_idx = (current_idx + chunk_size).min(text_content.len());
        let mut actual_end = end_idx;
        while !text_content.is_char_boundary(actual_end) {
            actual_end += 1;
        }
        batch_texts.push(text_content[current_idx..actual_end].to_string());
        current_idx = actual_end;
    }

    let batch_bytes: usize = batch_texts.iter().map(|s| s.len()).sum();

    println!("[Tokenizer Test] Токенизируем БАТЧ в МНОГОПОТОЧНОМ режиме...");
    println!("  |-> Всего строк на обработку: {}", batch_texts.len());
    println!(
        "  |-> Средний размер одной строки: {:.2} МБ",
        (batch_bytes as f64 / 64.0) / (1024.0 * 1024.0)
    );

    let start_parallel = Instant::now();
    let mut parallel_results = vec![Vec::new(); batch_texts.len()];

    std::thread::scope(|scope| {
        let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let t_chunk_size = (batch_texts.len() + num_threads - 1) / num_threads;

        let tokenizer_ref = &tokenizer;
        let batch_ref = &batch_texts;
        let mut results_chunks = parallel_results.chunks_mut(t_chunk_size);
        let mut text_chunks = batch_ref.chunks(t_chunk_size);

        for t_idx in 0..num_threads {
            if let (Some(r_chunk), Some(s_chunk)) = (results_chunks.next(), text_chunks.next()) {
                scope.spawn(move || {
                    let global_start = t_idx * t_chunk_size;

                    for (local_idx, text_line) in s_chunk.iter().enumerate() {
                        let line_id = global_start + local_idx;

                        println!(
                            "    [Поток #{}] Старт строки {} (Размер: {:.2} МБ)...",
                            t_idx,
                            line_id,
                            text_line.len() as f64 / (1024.0 * 1024.0)
                        );

                        let line_start = Instant::now();
                        r_chunk[local_idx] = tokenizer_ref.encode(text_line);

                        println!(
                            "    [Поток #{}] Готово строка {} за {:.2?}. Получено токенов: {}",
                            t_idx,
                            line_id,
                            line_start.elapsed(),
                            r_chunk[local_idx].len()
                        );
                    }
                });
            }
        }
    });

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
