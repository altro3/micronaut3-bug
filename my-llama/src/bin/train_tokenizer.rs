use memmap2::Mmap;
use std::fs::File;
use std::io::{BufReader, Error, ErrorKind};
use std::path::Path;
use std::time::Instant;

use my_llama::tokenizer::factory::types::QwenJsonModel;
use my_llama::tokenizer::trainer::bpe_trainer::BpeTrainer;
use my_llama::tokenizer::trainer::config::TrainerConfig;
use my_llama::tokenizer::trainer::utils::TrainerUtils;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("================ [СТАРТ РЕАКТИВНОЙ СБОРКИ BPE СЛОВАРЯ] ================");

    let input_text_path = "data/train/corpus.txt";
    let output_json_path = "data/train/my_qwen_model.json";

    if !Path::new(input_text_path).exists() {
        println!("[ОШИБКА] Обучающий текст не найден по пути: {}. Положи туда файл!", input_text_path);
        return Ok(());
    }

    println!("[ИНИЦИАЛИЗАЦИЯ] Проецирую обучающий корпус в виртуальную память через memmap2...");
    let file = File::open(input_text_path)?;
    let mmap_text = unsafe { Mmap::map(&file)? };
    let text_content = std::str::from_utf8(&mmap_text).map_err(|e| Error::new(ErrorKind::InvalidData, format!("Файл не в UTF-8: {:?}", e)))?;

    let file_size_mb = text_content.len() as f64 / 1024.0 / 1024.0;
    println!("  ├── Размер файла корпуса: {:.2} МБ", file_size_mb);

    let target_vocab_size = 128000;
    let config = TrainerConfig {
        batch_size: 256,
        initial_table_size: 524288,
        io_buffer_size: 4 * 1024 * 1024,
        start_token_id: 256,
        num_threads: 16,
        local_map_capacity: 32768,
        delta_map_capacity: 8192,
        position_buffer_capacity: 131072,
        index_rebuild_interval: 32,
    };

    println!("  ├── Целевой размер словаря (Vocab Size): {}", target_vocab_size);
    println!("  └── Количество воркеров (CPU Cores)    : {}", config.num_threads);
    println!("======================================================================");

    let trainer = BpeTrainer::new(target_vocab_size, config);

    let start_time = Instant::now();
    trainer.train(text_content, output_json_path)?;
    let total_elapsed = start_time.elapsed();

    let speed_mb_s = file_size_mb / total_elapsed.as_secs_f64();

    println!("\n======================================================================");
    println!("[УСПЕХ] Пайплайн обучения словаря Qwen полностью завершен!");
    println!("|-> Результат сохранен в : {}", output_json_path);
    println!("|-> Итоговое время сборки: {:?}", total_elapsed);
    println!("|-> Производительность пайплайна: {:.2} МБ/с", speed_mb_s);
    println!("======================================================================");

    println!("\n[АНАЛИЗ] Загрузка сгенерированного JSON для верификации кириллицы...");
    let check_file = File::open(output_json_path)?;
    let reader = BufReader::new(check_file);
    let model_data: QwenJsonModel = serde_json::from_reader(reader)?;
    let vocab = &model_data.model.vocab;

    let mut sorted_vocab: Vec<(&String, &u32)> = vocab.iter().collect();
    sorted_vocab.sort_unstable_by_key(|&(s, _)| std::cmp::Reverse(s.len()));

    println!("\n ТОП-30 САМЫХ ДЛИННЫХ СОБРАННЫХ ТОКЕНОВ (Проверка качества сжатия):");
    println!("----------------------------------------------------------------------");
    println!(" {:<10} | {:<30} | {}", "ID", "Qwen Строка (Сырая)", "Реальный Текст (Декодирован)");
    println!("----------------------------------------------------------------------");

    for (qwen_str, id_ref) in sorted_vocab.iter().take(30) {
        let id = **id_ref;
        let raw_bytes = TrainerUtils::qwen_string_to_bytes(qwen_str);
        let real_text = String::from_utf8_lossy(&raw_bytes).into_owned();

        let display_qwen = if qwen_str.chars().count() > 28 {
            format!("{}...", qwen_str.chars().take(25).collect::<String>())
        } else {
            qwen_str.to_string()
        };
        let display_real = if real_text.chars().count() > 28 {
            format!("{}...", real_text.chars().take(25).collect::<String>())
        } else {
            real_text
        };

        println!(" {:<10} | {:<30} | {}", id, display_qwen, display_real);
    }
    println!("----------------------------------------------------------------------\n");

    Ok(())
}
