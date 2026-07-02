use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use my_llama::tokenizer::trainer::{BpeTrainer, TrainerConfig};
use std::hint::black_box;
use std::time::Duration;

fn bench_bpe_trainer(c: &mut Criterion) {
    let mut group = c.benchmark_group("BpeTrainer_Extreme_MapReduce");

    // Тяжелый датасет объемом ~270 МБ
    let base_sample = "Привет мир! Это экстремально быстрый ии на Rust 2026 без компромиссов и глупых заглушек. Разгоняем параллельный MapReduce конвейер на полную мощность ядра! ";
    let iterations = 2_000_000;
    let mut training_text = String::with_capacity(base_sample.len() * iterations);
    for _ in 0..iterations {
        training_text.push_str(base_sample);
    }

    let text_bytes_len = training_text.len();
    group.throughput(Throughput::Bytes(text_bytes_len as u64));

    group.sample_size(10);
    group.measurement_time(Duration::from_secs(40));

    let config = TrainerConfig {
        batch_size: 512,
        initial_table_size: 524288,
        io_buffer_size: 4 * 1024 * 1024,
        start_token_id: 256,
    };

    let vocab_target_size = 2000;
    let trainer = BpeTrainer::new(vocab_target_size, config);

    #[cfg(target_os = "windows")]
    let dev_null = "NUL";
    #[cfg(not(target_os = "windows"))]
    let dev_null = "/dev/null";

    group.bench_with_input(
        BenchmarkId::new("train_vocab_heavy", format!("{:.1} MB", text_bytes_len as f64 / 1_048_576.0)),
        &training_text,

        |b, text| {
            b.iter(|| {
                let result = trainer.train(black_box(text), black_box(dev_null));
                let _ = black_box(result);
            });
        },
    );

    group.finish();
}

criterion_group!(benches, bench_bpe_trainer);
criterion_main!(benches);
