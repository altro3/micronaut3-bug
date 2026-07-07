use my_llama::tokenizer::bpe::context::TokenizationContext;
use my_llama::tokenizer::bpe::pipeline::TokenizerPipeline;
use my_llama::tokenizer::dfa::compiler::DfaCompiler;
use my_llama::tokenizer::dfa::runtime::FlatDfaRuntime;
use my_llama::tokenizer::factory::compiler::DictCompiler;
use my_llama::tokenizer::trainer::bpe_trainer::BpeTrainer;
use my_llama::tokenizer::trainer::config::TrainerConfig;
use my_llama::tokenizer::trainer::utils::TrainerUtils;
use my_llama::tokenizer::BpeTokenizer;
use sonic_rs::{from_reader, JsonContainerTrait, JsonValueTrait, Value};
use std::collections::HashMap;
use std::fs::{create_dir_all, File};
use std::io::{BufReader, Read};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n================ [СТАРТ ЖЕСТКОГО ИНТЕГРАЦИОННОГО ТЕСТА КИРИЛЛИЦЫ] ================");

    let test_corpus = "Яндекс — российская компания";
    let test_json_path = "data/train/test_ru_model.json";
    let dfa_trans_path = "data/train/qwen_dfa_trans.bin";
    let dfa_accept_path = "data/train/qwen_dfa_accept.bin";

    create_dir_all("data/train")?;

    let config = TrainerConfig { vocab_size: 268, start_token_id: 256, num_threads: 1, ..Default::default() };

    let total_merges = config.vocab_size - 256;
    println!("  ├── Текст для обучения  : \"{}\"", test_corpus);
    println!("  ├── Размер алфавита (база) : 256 токенов");
    println!("  ├── Plan слияний    : {} итераций", total_merges);
    println!("  └── Целевой Vocab Size    : {}", config.vocab_size);
    println!("==================================================================================");

    let trainer = BpeTrainer::new(config);
    trainer.train(test_corpus, test_json_path)?;

    println!("\n[АНАЛИЗ] Верификация собранного тестового JSON...");
    let check_file = File::open(test_json_path)?;
    let reader = BufReader::new(check_file);
    let model_data: Value = from_reader(reader)?;

    let vocab = model_data["model"]["vocab"].as_object().expect("Словарь vocab отсутствует в JSON");

    let mut reverse_vocab = HashMap::new();
    for (qwen_str, id_val) in vocab {
        let id = id_val.as_u64().unwrap() as u32;
        let bytes = TrainerUtils::qwen_string_to_bytes(qwen_str);
        reverse_vocab.insert(id, bytes);
    }

    println!("\n================== [СПИСОК ВСЕХ СОБРАННЫХ ВЫСШИХ ТОКЕНОВ (256..400)] ==================");
    let mut sorted_vocab: Vec<(&u32, &Vec<u8>)> = reverse_vocab.iter().collect();
    sorted_vocab.sort_unstable_by_key(|&(id, _)| id);

    for (id, bytes) in sorted_vocab {
        if *id >= 256 {
            let decoded_text = String::from_utf8_lossy(bytes).into_owned();
            println!("  Токен ID: {} | Декодирован как: '{}'", id, decoded_text.escape_debug());
        }
    }
    println!("==================================================================================\n");

    println!("\n[ДИАГНОСТИКА ТОКЕНИЗАЦИИ - ЕДИНИЧНЫЙ ТЕСТ]");
    let test_sentence = "Яндекс — российская компания";
    println!("  Входной текст: {:?}", test_sentence);

    let compiled_vocab = DictCompiler::compile_from_json(test_json_path)?;

    let bpe_tokenizer = BpeTokenizer::new(
        &compiled_vocab.raw_pairs,
        compiled_vocab.byte_fallback,
        compiled_vocab.eos_token_id,
        compiled_vocab.vocab_size,
        &compiled_vocab.vocab_compiled_tokens,
        compiled_vocab.trie_nodes,
        compiled_vocab.trie_root_offsets,
    );

    if !Path::new(dfa_trans_path).exists() || !Path::new(dfa_accept_path).exists() {
        println!("[РАНТАЙМ-БЕНЧМАРК] Кэш таблиц переходов не найден. Запускаю разовую компиляцию DFA...");
        DfaCompiler::compile_qwen_dfa(&compiled_vocab.extracted_regex, dfa_trans_path, dfa_accept_path)?;
    } else {
        println!("[РАНТАЙМ-БЕНЧМАРК] Обнаружен готовый кэш DFA. Загружаю предкомпилированные таблицы...");
    }

    // --- ИСПРАВЛЕНИЕ: Высокопроизводительное чтение бинарных DFA таблиц ---
    let mut trans_file = File::open(dfa_trans_path)?;
    let mut accept_file = File::open(dfa_accept_path)?;

    let mut trans_bytes = Vec::new();
    let mut accept_bytes = Vec::new();

    trans_file.read_to_end(&mut trans_bytes)?;
    accept_file.read_to_end(&mut accept_bytes)?;

    let dfa_runtime = FlatDfaRuntime::from_binary_dump(0, &trans_bytes, &accept_bytes);
    // ---------------------------------------------------------------------

    let pipeline = TokenizerPipeline::new(bpe_tokenizer, dfa_runtime);

    let single_ctx = TokenizationContext::new(compiled_vocab.vocab_size, 1024);
    let test_batch = vec![test_sentence];
    let test_encoded = pipeline.encode_parallel(&test_batch, &mut [single_ctx]);

    if let Some(tokens) = test_encoded.first() {
        println!("  Полученные токены (ID): {:?}", tokens);
        println!("  Потокеновый разбор декодером:");

        for &t_id in tokens {
            let id = t_id as usize;
            let mut raw_token_bytes = Vec::new();

            if id < pipeline.tokenizer.vocab_offsets_flat.len() {
                let packed = pipeline.tokenizer.vocab_offsets_flat[id];
                if packed != 0 {
                    let offset = (packed >> 32) as usize;
                    let length = (packed & 0xFFFFFFFF) as usize;
                    raw_token_bytes = pipeline.tokenizer.vocab_bytes_flat[offset..offset + length].to_vec();
                } else if id < 512 {
                    let b = pipeline.tokenizer.id_to_byte[id];
                    if b != 0xFF {
                        raw_token_bytes.push(b);
                    }
                }
            }

            let lossy_string = String::from_utf8_lossy(&raw_token_bytes).into_owned();

            println!(
                "    ID: {:6} -> Реальные байты в словаре: {:X?} -> Попытка отображения: {:?}",
                t_id, raw_token_bytes, lossy_string
            );
        }

        let final_decoded = pipeline.tokenizer.decode(tokens);
        println!("  Финальная сборка строки декодером: {:?}", final_decoded);

        assert_eq!(
            test_sentence, final_decoded,
            "КРИТИЧЕСКИЙ БАГ: Исходный текст и декодированный не совпадают!"
        );
        println!("  ├── [УСПЕХ] Строка восстановлена байт-в-байт без искажений.");
    }
    println!("==================================================\n");

    println!("[УСПЕХ] Тест полностью завершен. Движок успешно прожевал сложный литературный текст!");
    Ok(())
}
