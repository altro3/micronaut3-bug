use my_llama::tokenizer::factory::TokenizerFactory;
use std::fs;
use std::time::Instant;

fn main() -> std::io::Result<()> {
    // Выделяем 32 МБ стека на всякий случай, хотя Boxer-накатка фабрики защищает Windows
    let stack_size = 32 * 1024 * 1024;

    let handle = std::thread::Builder::new()
        .name("ultra-runtime".to_string())
        .stack_size(stack_size)
        .spawn(|| run_pure_tokenizer_benchmark())?;

    handle.join().unwrap()
}

fn run_pure_tokenizer_benchmark() -> std::io::Result<()> {
    let input_path = "data/input1.txt";
    let model_path = "data/qwen_model.json";

    // 1. ПРОВЕРКА ФАЙЛОВ
    assert!(
        std::path::Path::new(input_path).exists(),
        "КРИТИЧЕСКАЯ ОШИБКА: Положите ваш 750-мегабайтный файл кириллицы по пути data/input.txt"
    );
    assert!(
        std::path::Path::new(model_path).exists(),
        "КРИТИЧЕСКИЙ СБОЙ: Положите оригинальный файл Qwen tokenizer.json по пути data/qwen_model.json"
    );

    println!("[РАНТАЙМ-БЕНЧМАРК] Считываю исходный текстовый дамп в память...");
    let text_content = fs::read_to_string(input_path)?;
    let total_bytes = text_content.len();
    println!(
        "[РАНТАЙМ-БЕНЧМАРК] Размер исходного текста: {:.2} МБ ({:.2} ГБ)",
        total_bytes as f64 / (1024.0 * 1024.0),
        total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    // 2. ЗАГРУЗКА МОДЕЛИ QWEN ЧЕРЕЗ НАШУ SIMD ФАБРИКУ (0 байт на стеке)
    println!("[РАНТАЙМ-БЕНЧМАРК] Загружаем промышленную модель Qwen через SIMD-фабрику...");
    let start_factory = Instant::now();
    let tokenizer = TokenizerFactory::from_file(model_path)?;
    println!("[РАНТАЙМ-БЕНЧМАРК] Модель скомпилирована фабрикой за: {:?}", start_factory.elapsed());

    // 3. НАРЕЗКА БАТЧА СТРОГО ПО ГРАНИЦАМ СИМВОЛОВ UTF-8
    println!("[РАНТАЙМ-БЕНЧМАРК] Нарезаю 750 МБ текста на 64 параллельных потоковых чанка...");
    let mut batch_texts = Vec::with_capacity(64);
    let chunk_size = text_content.len() / 64;
    let mut current_idx = 0;

    for _ in 0..64 {
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

    // 4. ФАЗА РАЗГОНА И ПРОГРЕВА КЭША (WARM-UP)
    println!("[РАНТАЙМ-БЕНЧМАРК] Включаем турбо-буст CPU и прогреваем L1/L2/L3 кэши (Warm-up)...");
    for _ in 0..3 {
        std::hint::black_box(tokenizer.encode_parallel(&batch_texts));
    }

    println!("[РАНТАЙМ-БЕНЧМАРК] Запускаю параллельное кодирование кириллицы на P-ядрах...");
    let start_parallel = Instant::now();
    let parallel_results = tokenizer.encode_parallel(&batch_texts);
    let duration_parallel = start_parallel.elapsed();

    // Считаем метрики
    let total_tokens: usize = parallel_results.iter().map(|v| v.len()).sum();
    let speed_parallel_mib = (batch_bytes as f64 / 1024.0 / 1024.0) / duration_parallel.as_secs_f64();

    println!("\n================ [ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ СИНХРОНИЗАЦИИ] ================");
    println!("|-> Обработано исходных данных       : {} байт", batch_bytes);
    println!("|-> Суммарно сгенерировано токенов   : {} шт.", total_tokens);
    println!("|-> Чистый коэффициент BPE-сжатия    : {:.2}x", batch_bytes as f64 / total_tokens as f64);
    println!("|-> Чистое время токенизации батча   : {:?}", duration_parallel);
    println!(
        "|-> Скорость параллельного инференса  : {:.2} MiB/s ({:.2} GiB/s)",
        speed_parallel_mib,
        speed_parallel_mib / 1024.0
    );
    println!("======================================================================");

    // 6. ПРОВЕРКА СТАБИЛЬНОСТИ ОБРАТНОГО ДЕКОДЕРА ЧЕРЕЗ DECODER.RS
    println!("[РАНТАЙМ-БЕНЧМАРК] Тестирую in-place демаппинг декодера на первом чанке...");
    if let Some(first_tokens) = parallel_results.first() {
        let start_decode = Instant::now();
        let decoded_sample = tokenizer.decode(first_tokens);
        let duration_decode = start_decode.elapsed();

        assert!(!decoded_sample.is_empty(), "Критическая ошибка: Декодер вернул пустоту!");
        println!("|-> Декодер стабилен. Время восстановления одного чанка (~11 МБ): {:?}", duration_decode);
    }

    Ok(())
}
