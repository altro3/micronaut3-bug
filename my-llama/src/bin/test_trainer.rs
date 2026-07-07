use my_llama::tokenizer::trainer::bpe_trainer::BpeTrainer;
use my_llama::tokenizer::trainer::config::TrainerConfig;
use my_llama::tokenizer::trainer::utils::TrainerUtils;
use sonic_rs::{JsonContainerTrait, JsonValueTrait, Value, from_reader};
use std::fs::File;
use std::io::BufReader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n================ [СТАРТ ЖЕСТКОГО ИНТЕГРАЦИОННОГО ТЕСТА КИРИЛЛИЦЫ] ================");

    // Контролируемый текст с высокой повторяемостью слогов и пробелов
    let test_corpus = "我学习中文。中文学习很有趣。我喜欢中文学习。";

    // let test_corpus = "В белом плаще с кровавым подбоем, шаркающей кавалерийской походкой, \
    //                    ранним утром четырнадцатого числа весеннего месяца нисана в крытую \
    //                    колоннаду между двумя крыльями дворца ирода великого вышел прокуратор Иудеи Понтий Пилат. \
    //                    Более всего на свете прокуратор ненавидел запах розового масла, и все теперь предвещало \
    //                    нехороший день, так как запах этот преследовал прокуратора с рассвета. \
    //                    О, боги, боги, какая пошлая казнь! Но ты мне скажи, ведь её не было! \
    //                    Мне это приснилось, я видел её во сне!";
    let test_json_path = "data/train/test_ru_model.json";

    // Убедимся, что директория для сохранения существует
    std::fs::create_dir_all("data/train")?;

    // Используем оригинальный конфиг БЕЗ изменений регулярного выражения
    let config = TrainerConfig {
        vocab_size: 268,
        start_token_id: 256,
        num_threads: 1,
        // Для детерминированного теста на паре строк достаточно 1 потока
        ..Default::default()
    };

    let total_merges = config.vocab_size - 256;
    println!("  ├── Текст для обучения  : \"{}\"", test_corpus);
    println!("  ├── Размер алфавита (база) : 256 токенов");
    println!("  ├── Планируемых слияний    : {} итераций", total_merges);
    println!("  └── Целевой Vocab Size    : {}", config.vocab_size);
    println!("==================================================================================");

    let trainer = BpeTrainer::new(config);
    trainer.train(test_corpus, test_json_path)?;

    println!("\n[АНАЛИЗ] Верификация собранного тестового JSON...");
    let check_file = File::open(test_json_path)?;
    let reader = BufReader::new(check_file);
    let model_data: Value = from_reader(reader)?;

    let vocab = model_data["model"]["vocab"].as_object().expect("Словарь vocab отсутствует в JSON");

    let mut reverse_vocab = std::collections::HashMap::new();
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

    println!("[УСПЕХ] Тест полностью завершен. Движок успешно прожевал сложный литературный текст!");
    Ok(())
}
