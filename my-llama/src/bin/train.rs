use my_llama::cuda::CudaStream;
use my_llama::init_framework;
use my_llama::models::factory::ModelFactory;
use std::io::{Error, ErrorKind};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_framework();

    let stream = CudaStream::new();
    let config_path = "models/llama3_8b/config.json";
    let weights_path = "models/llama3_8b/model.safetensors";

    let batch_size = 4;
    let is_training = true;

    let mut training_graph = ModelFactory::create_from_config(config_path, weights_path, batch_size, is_training, &stream)?;

    let seq_len = 512;
    let host_input_tokens = vec![128000u32; batch_size * seq_len];
    let host_target_tokens = vec![1534i32; batch_size * seq_len];
    let mut host_losses = vec![0.0f32; batch_size * seq_len];

    println!("\n[СТАРТ] Запуск асинхронного конвейера обучения (Stream-Pipelined Loop)...");
    let total_train_timer = Instant::now();
    let iterations = 100;

    for _ in 1..=iterations {
        training_graph.load_targets(&host_target_tokens, &stream);

        let losses_gpu_buffer = training_graph
            .forward(&host_input_tokens, &stream)
            .map_err(|e| Error::new(ErrorKind::Other, e))?;

        losses_gpu_buffer.copy_to_host_slice(&mut host_losses, &stream);
    }

    println!("[КОНВЕЙЕР] Ожидание завершения очереди задач на GPU...");
    stream.synchronize();

    let elapsed = total_train_timer.elapsed();
    let average_loss: f32 = host_losses.iter().sum::<f32>() / host_losses.len() as f32;

    println!("\n[ОБУЧЕНИЕ ЗАВЕРШЕНО]");
    println!("|-> Успешно выполнено шагов: {}", iterations);
    println!("|-> Финальный средний Loss: {:.4}", average_loss);
    println!("|-> Полное время процесса: {:.2?}", elapsed);
    println!("|-> Среднее скорость: {:.2?} на один полный шаг (Fwd+Bwd+AdamW)", elapsed / iterations);

    Ok(())
}
