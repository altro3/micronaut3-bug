use memmap2::Mmap;
use std::fs::File;
use std::io::{Error, ErrorKind};
use std::path::Path;
use std::time::Instant;

use my_llama::tokenizer::trainer::bpe_trainer::BpeTrainer;
use my_llama::tokenizer::trainer::config::TrainerConfig;

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
        batch_size: 256,                 // Оптимальный шаг слияний для кириллицы
        initial_table_size: 524288,      // Размер глобальной хэш-таблицы
        io_buffer_size: 4 * 1024 * 1024, // Буфер BufWriter для сброса JSON
        start_token_id: 256,             // ID первого нового токена после байтового алфавита
        num_threads: 16,                 // Потоки строго под 16 физических ядер процессора
        local_map_capacity: 16384,       // Локальная мапа (~256 КБ) — садится в L2-кэш каждого ядра
        delta_map_capacity: 4096,        // Мапа дельт пар — не реаллоцируется во время шага
        position_buffer_capacity: 65536, // Стартовый буфер очагов мутации плоского корпуса
        index_rebuild_interval: 16,      // Шаг ленивой дефрагментации инвертированного индекса
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

    Ok(())
}
