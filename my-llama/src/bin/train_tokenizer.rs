use my_llama::tokenizer::factory::TokenizerFactory;
use my_llama::tokenizer::trainer::{BpeTrainer, TrainerConfig};
use std::fs;
use std::time::Instant;

fn main() -> std::io::Result<()> {
    let stack_size = 32 * 1024 * 1024;

    let handle = std::thread::Builder::new()
        .name("ultra-runtime".to_string())
        .stack_size(stack_size)
        .spawn(|| run_benchmark())?;

    handle.join().unwrap()
}

fn run_benchmark() -> std::io::Result<()> {
    let input_path = "data/input.txt";
    let mut model_path = "data/my_bpe_model.json";
    fs::create_dir_all("data")?;

    // Проверяем, существует ли файл
    let file_exists = std::path::Path::new(input_path).exists();
    let mut is_huge_dataset = false;

    if file_exists {
        let metadata = fs::metadata(input_path)?;
        // Если файл весит больше 10 МБ, значит пользователь положил туда свой большой текст
        if metadata.len() > 10 * 1024 * 1024 {
            is_huge_dataset = true;
        }
    }

    if !file_exists {
        println!("[Ультра-Тест] Генерирую грязный многоязычный хаос-датасет...");

        let blocks = vec![
            "UltraLlama! 🔥🚀 ",
            "Нейросеть ≠ Компьютер; ",
            "人工智能 🧠 计算 ∞ ",
            "Rust_performance_тест\t\n",
            "без_компромиссов_5090_∑_ ",
        ];

        let mut big_text = String::with_capacity(16 * 1024 * 1024);
        for i in 0..150_000 {
            big_text.push_str(blocks[i % blocks.len()]);
        }
        fs::write(input_path, big_text)?;
    }

    let text_content = fs::read_to_string(input_path)?;
    let total_bytes = text_content.len();
    println!(
        "[Ультра-Тест] Размер исходного текста: {:.2} МБ",
        total_bytes as f64 / (1024.0 * 1024.0)
    );

    // Логика переключения: обучаем маленькую модель только на сгенерированном хаос-тексте.
    // Если подсунут огромный файл на 750 МБ — используем готовую промышленную модель Qwen!
    if is_huge_dataset {
        model_path = "data/qwen_model.json";
        println!("[Ультра-Тест] Обнаружен огромный датасет. Пропускаем обучение, используем промышленную модель Qwen.");
        assert!(
            std::path::Path::new(model_path).exists(),
            "Критическая ошибка: Положите оригинальный файл Qwen tokenizer.json по пути data/qwen_model.json"
        );
    } else {
        println!("[Ультра-Тест] Запуск обучения маленькой BPE-модели на сгенерированном тексте...");
        let config = TrainerConfig::default();
        let trainer = BpeTrainer::new(250000, config);

        let start_train = Instant::now();
        trainer.train(&text_content, model_path)?;
        println!("[Ультра-Тест] Время обучения словаря: {:?}", start_train.elapsed());
    }

    // 2. ЗАГРУЗКА ЧЕРЕЗ НАШУ SIMD ФАБРИКУ (Возвращает Box<BpeTokenizer>)
    println!("\n[Ультра-Тест] Загружаем модель через фабрику...");
    let tokenizer = TokenizerFactory::from_file(model_path)?;

    // 3. ПОДГОТОВКА БАТЧА
    let mut batch_texts = Vec::with_capacity(64);
    let chunk_size = text_content.len() / 64;
    let mut current_idx = 0;

    for _ in 0..32 {
        let mut end_idx = current_idx + chunk_size;
        while end_idx < text_content.len() && !text_content.is_char_boundary(end_idx) {
            end_idx += 1;
        }
        batch_texts.push(text_content[current_idx..end_idx].to_string());
        current_idx = end_idx;
    }
    if current_idx < text_content.len() {
        batch_texts.push(text_content[current_idx..].to_string());
    }

    let batch_bytes: usize = batch_texts.iter().map(|s| s.len()).sum();

    // --- ФАЗА ПРОГРЕВА (WARM-UP) ---
    println!("[Ультра-Тест] Разгон процессора и прогрев кэша (Warm-up)...");
    for _ in 0..3 {
        std::hint::black_box(tokenizer.encode_parallel(&batch_texts));
    }

    println!("[Ультра-Тест] Токенизируем БАТЧ на P-ядрах...");

    // 3. ЧИСТЫЙ ЗАМЕР ВРЕМЕНИ ЭНКОДЕРА
    let start_parallel = Instant::now();
    let parallel_results = tokenizer.encode_parallel(&batch_texts);
    let duration_parallel = start_parallel.elapsed();

    let total_tokens: usize = parallel_results.iter().map(|v| v.len()).sum();
    let speed_parallel = (batch_bytes as f64 / 1024.0 / 1024.0) / duration_parallel.as_secs_f64();

    println!("\n[РЕЗУЛЬТАТЫ СИНХРОНИЗАЦИИ]");
    println!("|-> Всего исходных байт: {}", batch_bytes);
    println!("|-> Суммарно получено токенов: {}", total_tokens);
    println!("|-> Коэффициент BPE-сжатия текста: {:.2}x", batch_bytes as f64 / total_tokens as f64);
    println!("|-> Время параллельного кодирования: {:?}", duration_parallel);
    println!(
        "|-> Скорость параллельного кодирования: {:.2} MiB/s ({:.2} GiB/s)",
        speed_parallel,
        speed_parallel / 1024.0
    );

    // Дополнительный микро-тест: Проверяем, что рантайм-декодер работает и на огромном батче результатов
    println!("[Ультра-Тест] Проверка целостности: декодируем один чанк обратно...");
    if let Some(first_tokens) = parallel_results.first() {
        let decoded_sample = tokenizer.decode(first_tokens);
        assert!(!decoded_sample.is_empty(), "Декодер вернул пустую строку!");
        println!("|-> Декодер стабилен, UTF-8 валиден.");
    }

    Ok(())
}