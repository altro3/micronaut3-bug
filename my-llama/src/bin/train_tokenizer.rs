use my_llama::tokenizer::bpe::context::TokenizationContext;
use my_llama::tokenizer::bpe::pipeline::TokenizerPipeline;
use my_llama::tokenizer::dfa::runtime::FlatDfaRuntime;
use my_llama::tokenizer::factory::compiler::DictCompiler;
use my_llama::tokenizer::trainer::{BpeTrainer, TrainerConfig};
use my_llama::tokenizer::BpeTokenizer;
use std::fs;
use std::path::Path;
use std::time::Instant;
use my_llama::tokenizer::dfa::compiler::DfaCompiler;

fn main() -> std::io::Result<()> {
    let stack_size = 32 * 1024 * 1024;
    std::thread::Builder::new()
        .name("ultra-runtime".to_string())
        .stack_size(stack_size)
        .spawn(|| run_benchmark())?
        .join()
        .unwrap()
}

fn run_benchmark() -> std::io::Result<()> {
    let input_path = "data/input.txt";
    let mut model_path = "data/my_bpe_model.json";

    let dfa_trans_path = "data/qwen_dfa_trans.bin";
    let dfa_accept_path = "data/qwen_dfa_accept.bin";

    fs::create_dir_all("data")?;

    let file_exists = Path::new(input_path).exists();
    let mut is_huge_dataset = false;

    if file_exists {
        let metadata = fs::metadata(input_path)?;
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
    println!("[Ультра-Тест] Размер исходного текста: {:.2} МБ", total_bytes as f64 / (1024.0 * 1024.0));

    if is_huge_dataset {
        model_path = "data/qwen_model.json";
        println!("[Ультра-Тест] Обнаружен огромный датасет. Используем промышленную модель Qwen.");
        assert!(Path::new(model_path).exists(), "Положите файл в data/qwen_model.json");
    } else {
        println!("[Ультра-Тест] Запуск обучения маленькой BPE-модели...");
        let config = TrainerConfig::default();
        let trainer = BpeTrainer::new(250000, config);
        let start_train = Instant::now();
        trainer.train(&text_content, model_path)?;
        println!("[Ультра-Тест] Время обучения словаря: {:?}", start_train.elapsed());
    }

    // 1. ЗАГРУЗКА И ДИНАМИЧЕСКИЙ РАЗБОР JSON МОДЕЛИ
    println!("\n[Ультра-Тест] Загружаем модель через DictCompiler...");
    let compiled_vocab = DictCompiler::compile_from_json(model_path)?;

    let bpe_tokenizer = BpeTokenizer::new(
        &compiled_vocab.raw_pairs,
        compiled_vocab.byte_fallback,
        compiled_vocab.eos_token_id,
        compiled_vocab.vocab_size, // ИСПРАВЛЕНО: Чтение обычного публичного поля
        &compiled_vocab.vocab_compiled_tokens,
    );

    // 2. УМНОЕ КЭШИРОВАНИЕ ДИНАМИЧЕСКОГО DFA
    if !Path::new(dfa_trans_path).exists() || !Path::new(dfa_accept_path).exists() {
        println!("[Ультра-Тест] Кэш таблиц переходов не найден. Запускаю разовую компиляцию DFA...");
        DfaCompiler::compile_qwen_dfa(&compiled_vocab.extracted_regex, dfa_trans_path, dfa_accept_path)?;
    }

    // Загружаем скомпилированные таблицы переходов напрямую с диска
    let trans_bytes = fs::read(dfa_trans_path)?;
    let accept_bytes = fs::read(dfa_accept_path)?;
    let dfa_runtime = FlatDfaRuntime::from_binary_dump(0, &trans_bytes, &accept_bytes);

    let pipeline = TokenizerPipeline::new(bpe_tokenizer, dfa_runtime);

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

    // 4. ПРЕДВЫДЕЛЕНИЕ КОНТЕКСТОВ ПОТОКОВ (Zero malloc)
    let num_threads = batch_texts.len();
    let mut contexts: Vec<TokenizationContext> = std::iter::repeat_with(|| TokenizationContext::new(compiled_vocab.vocab_size, chunk_size * 2))
        .take(num_threads)
        .collect();

    // --- ФАЗА ПРОГРЕВА (WARM-UP) ---
    println!("[Ультра-Тест] Разгон процессора и прогрев кэша (Warm-up)...");
    for _ in 0..2 {
        std::hint::black_box(pipeline.encode_parallel(&batch_texts, &mut contexts));
    }

    // 5. ЧИСТЫЙ ЗАМЕР ВРЕМЕНИ
    println!("[Ультра-Тест] Токенизируем БАТЧ на P-ядрах...");
    let start_parallel = Instant::now();
    let parallel_results = pipeline.encode_parallel(&batch_texts, &mut contexts);
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

    if let Some(first_tokens) = parallel_results.first() {
        let decoded_sample = pipeline.tokenizer.decode(first_tokens);
        assert!(!decoded_sample.is_empty(), "Декодер вернул пустую строку!");
        println!("|-> Декодер стабилен, UTF-8 валиден.");
    }

    Ok(())
}
