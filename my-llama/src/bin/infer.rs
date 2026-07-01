use io::{Error, ErrorKind};
use my_llama::cuda::CudaStream;
use my_llama::models::factory::ModelFactory;
use std::io::{self, Write};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    my_llama::init_framework();

    println!("[INIT] Запуск промышленного инференс-движка на монолитной арене VRAM...");

    let stream = CudaStream::new();
    let config_path = "models/llama3_8b/config.json";
    let weights_path = "models/llama3_8b/model.safetensors";
    let batch_size = 1;
    let is_training = false;

    let timer = Instant::now();
    let mut model_graph = match ModelFactory::create_from_config(config_path, weights_path, batch_size, is_training, &stream) {
        Ok(graph) => graph,
        Err(e) => {
            eprintln!("[CRITICAL] Ошибка инициализации графа фабрикой: {}", e);
            std::process::exit(1);
        }
    };
    println!("[INIT] Граф успешно скомпилирован и загружен в VRAM за {:.2?}", timer.elapsed());

    let mut current_token_id: u32 = 128000;
    let mut input_tokens = vec![current_token_id];

    println!("\n[ЧАТ ЗАПУЩЕН] Начинаем генерацию текста (нажмите Ctrl+C для выхода):");
    print!("Модель: ");
    io::stdout().flush()?;

    let mut generated_tokens_count = 0;
    let total_generation_timer = Instant::now();

    for _ in 0..100 {
        let step_timer = Instant::now();

        let logits_gpu_buffer = model_graph
            .forward(&input_tokens, &stream)
            .map_err(|e| Error::new(ErrorKind::Other, e))?;

        let vocab_size = model_graph.vocab_size();
        let mut host_logits = vec![0.0f32; vocab_size];

        logits_gpu_buffer.copy_to_host_slice(&mut host_logits, &stream);

        stream.synchronize();

        let mut max_idx = 0;
        let mut max_val = host_logits[0];
        for (idx, &val) in host_logits.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = idx;
            }
        }

        current_token_id = max_idx as u32;
        input_tokens[0] = current_token_id;
        generated_tokens_count += 1;

        print!("[{}] ", current_token_id);
        io::stdout().flush()?;

        println!(" ({:.2?}/token)", step_timer.elapsed());
    }

    let elapsed = total_generation_timer.elapsed();
    let tps = generated_tokens_count as f64 / elapsed.as_secs_f64();
    println!("\n\n[ГЕНЕРАЦИЯ ЗАВЕРШЕНА]");
    println!("|-> Всего сгенерировано токенов: {}", generated_tokens_count);
    println!("|-> Чистое время генерации: {:.2?}", elapsed);
    println!("|-> Промышленная скорость инференса: {:.2} токенов/сек (t/s)", tps);

    Ok(())
}
