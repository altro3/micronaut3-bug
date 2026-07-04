use my_llama::tokenizer::dfa::runtime::FlatDfaRuntime;
use my_llama::tokenizer::trainer::{BpeTrainer, TrainerConfig};
use std::fs;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("================ [СТАРТ СБОРКИ BPE СЛОВАРЯ] ================");

    // 1. Укажи пути к твоему обучающему тексту и файлу кэша DFA
    let input_text_path = "data/train/corpus.txt"; // Твой файл с текстом для обучения
    let dfa_trans_path = "data/qwen_dfa_trans.bin"; // Бинарник матрицы переходов твоей регулярки
    let output_json_path = "data/train/my_qwen_model.json"; // Куда сохранить готовый словарь

    // Проверяем наличие исходного текста
    if !std::path::Path::new(input_text_path).exists() {
        println!(
            "[ОШИБКА] Обучающий текст не найден по пути: {}. Положи туда любой текстовый файл!",
            input_text_path
        );
        return Ok(());
    }

    // 2. Инициализируем FlatDfaRuntime для честной нарезки Qwen-чанков
    println!("[ИНИЦИАЛИЗАЦИЯ] Загружаю матрицу переходов DFA...");
    let trans_bytes =
        fs::read(dfa_trans_path).map_err(|_| std::io::Error::new(std::io::ErrorKind::NotFound, "Матрица DFA регулярки не найдена! Проверь пути."))?;
    let dfa_splitter = FlatDfaRuntime::from_binary_dump(0, &trans_bytes, &[]);

    // 3. Читаем весь текст корпуса в оперативную память через mmap или string
    println!("[ИНИЦИАЛИЗАЦИЯ] Читаю обучающий корпус в RAM...");
    let text_content = fs::read_to_string(input_text_path)?;
    println!("  ├── Размер текстового файла: {:.2} МБ", text_content.len() as f64 / 1024.0 / 1024.0);

    // 4. Настраиваем конфигурацию обучения
    let target_vocab_size = 32000; // Целевой размер словаря (базовые 256 + 31744 мерджей)
    let config = TrainerConfig {
        batch_size: 512,            // Сколько топовых пар склеивать за одну итерацию
        initial_table_size: 262144, // Стартовый размер хэш-таблицы для воркеров
        io_buffer_size: 4 * 1024 * 1024,
        start_token_id: 256, // Первый свободный ID после ASCII/байт токенов
    };

    println!("  ├── Целевой размер словаря (Vocab Size): {}", target_vocab_size);
    println!("  └── Размер батча слияний (Batch Size)  : {}", config.batch_size);
    println!("======================================================================");

    // 5. Создаем тренера и запускаем параллельный конвейер
    let trainer = BpeTrainer::new(target_vocab_size, config, dfa_splitter);

    let start_time = Instant::now();
    trainer.train(&text_content, output_json_path)?;
    let total_elapsed = start_time.elapsed();

    println!("\n[УСПЕХ] Пайплайн обучения словаря Qwen полностью завершен!");
    println!("|-> Результат сохранен в : {}", output_json_path);
    println!("|-> Итоговое время сборки: {:?}", total_elapsed);
    println!("======================================================================");

    Ok(())
}
