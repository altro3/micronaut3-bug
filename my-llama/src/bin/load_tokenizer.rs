use memmap2::Mmap;
use my_llama::tokenizer::bpe::context::TokenizationContext;
use my_llama::tokenizer::bpe::pipeline::TokenizerPipeline;
use my_llama::tokenizer::dfa::compiler::DfaCompiler;
use my_llama::tokenizer::dfa::runtime::FlatDfaRuntime;
use my_llama::tokenizer::factory::compiler::DictCompiler;
use my_llama::tokenizer::BpeTokenizer;
use std::fs::File;
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
    let input_path = "data/input1_x4.txt";
    let model_path = "data/qwen_model.json";
    let dfa_trans_path = "data/qwen_dfa_trans.bin";
    let dfa_accept_path = "data/qwen_dfa_accept.bin";

    assert!(
        std::path::Path::new(input_path).exists(),
        "Положите 750-МБ файл кириллицы в data/input1.txt"
    );
    assert!(
        std::path::Path::new(model_path).exists(),
        "Положите qwen_model.json в data/qwen_model.json"
    );

    println!("[РАНТАЙМ-БЕНЧМАРК] Проецирую исходный текст через memmap2...");
    let file = File::open(input_path)?;
    let mmap_text = unsafe { Mmap::map(&file)? };

    let text_content = std::str::from_utf8(&mmap_text).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    let total_bytes = text_content.len();
    println!(
        "[РАНТАЙМ-БЕНЧМАРК] Размер проецированного текста: {:.2} МБ (Потребление RAM процесса: ~0 байт)",
        total_bytes as f64 / 1024.0 / 1024.0
    );

    println!("[РАНТАЙМ-БЕНЧМАРК] Загружаем промышленную модель Qwen через DictCompiler...");
    let start_factory = Instant::now();
    let compiled_vocab = DictCompiler::compile_from_json(model_path)?;

    let bpe_tokenizer = BpeTokenizer::new(
        &compiled_vocab.raw_pairs,
        compiled_vocab.byte_fallback,
        compiled_vocab.eos_token_id,
        compiled_vocab.vocab_size,
        &compiled_vocab.vocab_compiled_tokens,
        compiled_vocab.trie_nodes,
        compiled_vocab.trie_root_offsets,
    );

    if !std::path::Path::new(dfa_trans_path).exists() || !std::path::Path::new(dfa_accept_path).exists() {
        println!("[РАНТАЙМ-БЕНЧМАРК] Кэш таблиц переходов не найден. Запускаю разовую компиляцию DFA...");
        DfaCompiler::compile_qwen_dfa(&compiled_vocab.extracted_regex, dfa_trans_path, dfa_accept_path)?;
    } else {
        println!("[РАНТАЙМ-БЕНЧМАРК] Обнаружен готовый кэш DFA. Загружаю предкомпилированные таблицы...");
    }

    let trans_file = File::open(dfa_trans_path)?;
    let accept_file = File::open(dfa_accept_path)?;
    let mmap_trans = unsafe { Mmap::map(&trans_file)? };
    let mmap_accept = unsafe { Mmap::map(&accept_file)? };

    let dfa_runtime = FlatDfaRuntime::from_binary_dump(0, &mmap_trans, &mmap_accept);

    let pipeline = TokenizerPipeline::new(bpe_tokenizer, dfa_runtime);
    println!("[РАНТАЙМ-БЕНЧМАРК] Модель и DFA собраны in Pipeline за: {:?}", start_factory.elapsed());

    let mut batch_texts: Vec<&str> = Vec::with_capacity(65);
    let chunk_size = text_content.len() / 64;
    let mut current_idx = 0;
    for _ in 0..64 {
        let mut end_idx = current_idx + chunk_size;
        if end_idx >= text_content.len() {
            break;
        }
        while end_idx < text_content.len() && !text_content.is_char_boundary(end_idx) {
            end_idx += 1;
        }
        batch_texts.push(&text_content[current_idx..end_idx]);
        current_idx = end_idx;
    }
    if current_idx < text_content.len() {
        batch_texts.push(&text_content[current_idx..]);
    }
    let batch_bytes: usize = batch_texts.iter().map(|s| s.len()).sum();

    let num_threads = 16;
    let mut contexts: Vec<TokenizationContext> = std::iter::repeat_with(|| TokenizationContext::new(compiled_vocab.vocab_size, chunk_size / 2))
        .take(num_threads)
        .collect();

    println!("[РАНТАЙМ-БЕНЧМАРК] Запускаю параллельное кодирование кириллицы на ядрах...");
    let start_parallel = Instant::now();

    let parallel_results = pipeline.encode_parallel(&batch_texts, &mut contexts);
    let duration_parallel = start_parallel.elapsed();

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

    if let Some(first_tokens) = parallel_results.first() {
        let start_decode = Instant::now();
        let decoded_sample = pipeline.tokenizer.decode(first_tokens);
        let duration_decode = start_decode.elapsed();
        assert!(!decoded_sample.is_empty(), "Критическая ошибка: Декодер вернул пустоту!");
        println!("|-> Декодер стабилен. Время восстановления одного чанка: {:?}", duration_decode);
    }

    Ok(())
}
