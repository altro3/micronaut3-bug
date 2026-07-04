use memmap2::Mmap;
use my_llama::tokenizer::dfa::compiler::DfaCompiler;
use my_llama::tokenizer::dfa::runtime::FlatDfaRuntime;
use my_llama::tokenizer::trainer::{BpeTrainer, TrainerConfig};
use std::fs::File;
use std::io::{Error, ErrorKind};
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("================ [СТАРТ СБОРКИ BPE СЛОВАРЯ] ================");

    let input_text_path = "data/train/corpus.txt";
    let dfa_trans_path = "data/qwen_dfa_trans.bin";
    let dfa_accept_path = "data/qwen_dfa_accept.bin";
    let output_json_path = "data/train/my_qwen_model.json";

    if !Path::new(input_text_path).exists() {
        println!(
            "[ОШИБКА] Обучающий текст не найден по пути: {}. Положи туда любой текстовый файл!",
            input_text_path
        );
        return Ok(());
    }

    let cyrillic_regex = r" ?\p{L}+|\p{L}+| ?\p{N}+|[^\s\p{L}\p{N}]+|\s*[\r\n]+|\s+";

    if !Path::new(dfa_trans_path).exists() || !Path::new(dfa_accept_path).exists() {
        println!("[ИНИЦИАЛИЗАЦИЯ] Кэш таблиц переходов не найден. Запускаю компиляцию DFA...");
        DfaCompiler::compile_qwen_dfa(cyrillic_regex, dfa_trans_path, dfa_accept_path)?;
    }

    println!("[ИНИЦИАЛИЗАЦИЯ] Загружаю матрицу переходов и карту состояний DFA через memmap2...");

    let trans_file = File::open(dfa_trans_path)?;
    let mmap_trans = unsafe { Mmap::map(&trans_file)? };

    let accept_file = File::open(dfa_accept_path)?;
    let mmap_accept = unsafe { Mmap::map(&accept_file)? };

    println!("[ИНИЦИАЛИЗАЦИЯ] Загружаю матрицу переходов DFA...");
    let dfa_splitter = FlatDfaRuntime::from_binary_dump(0, &mmap_trans, &mmap_accept);

    println!("[ИНИЦИАЛИЗАЦИЯ] Проецирую обучающий корпус в виртуальную память через memmap2...");
    let file = File::open(input_text_path)?;
    let mmap_text = unsafe { Mmap::map(&file)? };

    let text_content = std::str::from_utf8(&mmap_text).map_err(|e| Error::new(ErrorKind::InvalidData, format!("Файл не в UTF-8: {:?}", e)))?;

    println!(
        "  ├── Размер проецированного файла: {:.2} МБ (RAM процесса: ~0 байт)",
        text_content.len() as f64 / 1024.0 / 1024.0
    );

    let target_vocab_size = 256000;
    let config = TrainerConfig {
        batch_size: 512,
        initial_table_size: 262144,
        io_buffer_size: 4 * 1024 * 1024,
        start_token_id: 256,
    };

    println!("  ├── Целевой размер словаря (Vocab Size): {}", target_vocab_size);
    println!("  └── Размер батча слияний (Batch Size)  : {}", config.batch_size);
    println!("======================================================================");

    let trainer = BpeTrainer::new(target_vocab_size, config, dfa_splitter);

    let start_time = Instant::now();
    trainer.train(text_content, output_json_path)?;
    let total_elapsed = start_time.elapsed();

    println!("\n[УСПЕХ] Пайплайн обучения словаря Qwen полностью завершен!");
    println!("|-> Результат сохранен в : {}", output_json_path);
    println!("|-> Итоговое время сборки: {:?}", total_elapsed);
    println!("======================================================================");

    Ok(())
}
