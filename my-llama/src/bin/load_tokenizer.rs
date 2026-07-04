use my_llama::tokenizer::bpe::context::TokenizationContext;
use my_llama::tokenizer::bpe::pipeline::TokenizerPipeline;
use my_llama::tokenizer::dfa::compiler::DfaCompiler;
use my_llama::tokenizer::dfa::runtime::FlatDfaRuntime;
use my_llama::tokenizer::factory::compiler::DictCompiler;
use my_llama::tokenizer::BpeTokenizer;
use std::fs;
use std::time::Instant;

fn main() -> std::io::Result<()> {
    let stack_size = 32 * 1024 * 1024;
    std::thread::Builder::new()
        .name("ultra-runtime".to_string())
        .stack_size(stack_size)
        .spawn(|| run_pure_tokenizer_benchmark())?
        .join()
        .unwrap()
}

fn run_pure_tokenizer_benchmark() -> std::io::Result<()> {
    let input_path = "data/input1.txt";
    let model_path = "data/qwen_model.json";
    let dfa_trans_path = "data/qwen_dfa_trans.bin"; // Твои бинарные дампы автомата
    let dfa_accept_path = "data/qwen_dfa_accept.bin";

    assert!(
        std::path::Path::new(input_path).exists(),
        "Положите 750-МБ файл кириллицы в data/input1.txt"
    );
    assert!(
        std::path::Path::new(model_path).exists(),
        "Положите qwen_model.json в data/qwen_model.json"
    );

    println!("[РАНТАЙМ-БЕНЧМАРК] Считываю исходный текстовый дамп в память...");
    let text_content = fs::read_to_string(input_path)?;
    let total_bytes = text_content.len();
    println!(
        "[РАНТАЙМ-БЕНЧМАРК] Размер исходного текста: {:.2} МБ",
        total_bytes as f64 / 1024.0 / 1024.0
    );

    // 1. ЗАГРУЗКА И ДИНАМИЧЕСКИЙ РАЗБОР JSON МОДЕЛИ
    println!("[РАНТАЙМ-БЕНЧМАРК] Загружаем промышленную модель Qwen через DictCompiler...");
    let start_factory = Instant::now();
    let compiled_vocab = DictCompiler::compile_from_json(model_path)?;

    let bpe_tokenizer = BpeTokenizer::new(
        &compiled_vocab.raw_pairs,
        compiled_vocab.byte_fallback,
        compiled_vocab.eos_token_id,
        compiled_vocab.vocab_size,
        &compiled_vocab.vocab_compiled_tokens,
    );

    // 2. АЛГОРИТМ УМНОГО КЭШИРОВАНИЯ ТАБЛИЦ DFA (Wordchipper-стиль):
    // Если бинарники уже лежат в data/, компилятор DFA ДАЖЕ НЕ ВКЛЮЧАЕТСЯ
    if !std::path::Path::new(dfa_trans_path).exists() || !std::path::Path::new(dfa_accept_path).exists() {
        println!("[РАНТАЙМ-БЕНЧМАРК] Кэш таблиц переходов не найден. Запускаю разовую компиляцию DFA...");
        DfaCompiler::compile_qwen_dfa(&compiled_vocab.extracted_regex, dfa_trans_path, dfa_accept_path)?;
    } else {
        println!("[РАНТАЙМ-БЕНЧМАРК] Обнаружен готовый кэш DFA. Загружаю предкомпилированные таблицы...");
    }

    // Мгновенная Zero-Copy десериализация кэша из файлов за доли миллисекунды
    let trans_bytes = fs::read(dfa_trans_path)?;
    let accept_bytes = fs::read(dfa_accept_path)?;
    let dfa_runtime = FlatDfaRuntime::from_binary_dump(0, &trans_bytes, &accept_bytes);

    // Собираем сквозной монолитный пайплайн инференса
    let pipeline = TokenizerPipeline::new(bpe_tokenizer, dfa_runtime);
    println!("[РАНТАЙМ-БЕНЧМАРК] Модель и DFA собраны в Pipeline за: {:?}", start_factory.elapsed());

    // 3. НАРЕЗКА БАТЧА ПО ГРАНИЦАМ СИМВОЛОВ UTF-8
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

    // 4. ПРЕДВЫДЕЛЕНИЕ КОНТЕКСТОВ ДЛЯ ПОТОКОВ (Zero malloc рантайм)
    let mut contexts: Vec<TokenizationContext> = std::iter::repeat_with(|| TokenizationContext::new(compiled_vocab.vocab_size, 2 * chunk_size))
        .take(64)
        .collect();

    // 5. ВАРМАП КЭША (Warm-up)
    println!("[РАНТАЙМ-БЕНЧМАРК] Прогреваем L1/L2/L3 кэши процессора (Warm-up)...");
    for _ in 0..2 {
        std::hint::black_box(pipeline.encode_parallel(&batch_texts, &mut contexts));
    }

    // 6. БЕНЧМАРК ПАРАЛЛЕЛЬНОГО КОРПУСА КИРИЛЛИЦЫ
    println!("[РАНТАЙМ-БЕНЧМАРК] Запускаю параллельное кодирование кириллицы на P-ядрах...");
    let start_parallel = Instant::now();
    let parallel_results = pipeline.encode_parallel(&batch_texts, &mut contexts);
    let duration_parallel = start_parallel.elapsed();

    // Метрики
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

    // 7. СТАБИЛЬНОСТЬ ОБРАТНОГО ДЕКОДЕРА
    if let Some(first_tokens) = parallel_results.first() {
        let start_decode = Instant::now();
        let decoded_sample = pipeline.tokenizer.decode(first_tokens);
        let duration_decode = start_decode.elapsed();
        assert!(!decoded_sample.is_empty(), "Критическая ошибка: Декодер вернул пустоту!");
        println!("|-> Декодер стабилен. Время восстановления одного чанка: {:?}", duration_decode);
    }

    Ok(())
}
