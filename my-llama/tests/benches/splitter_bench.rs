use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use my_llama::tokenizer::SimdSplitter;
use std::hint::black_box;

fn bench_simd_splitter(c: &mut Criterion) {
    let mut group = c.benchmark_group("SimdSplitter_V2_AVX2");

    // Генерируем тяжелый (~16 МБ) репрезентативный датасет для полной загрузки кэша L3 Arrow Lake
    let base_sample = "unbelievable_industrial_grade_artificial_intelligence_tokenizer_performance_test_on_rust ";
    let iterations = 200_000;
    let mut large_text = String::with_capacity(base_sample.len() * iterations);
    for _ in 0..iterations {
        large_text.push_str(base_sample);
    }

    let text_bytes_len = large_text.len();
    group.throughput(Throughput::Bytes(text_bytes_len as u64));

    // Создаем честный буфер u32 под новые ID токенов
    let mut ids_buffer = vec![0u32; text_bytes_len + 64];

    // Инициализируем статическую таблицу byte_fallback без внешних файлов
    let mut byte_fallback = [0u32; 256];
    for b in 0..=255 {
        byte_fallback[b] = b as u32;
    }

    group.bench_with_input(
        BenchmarkId::new("split_to_ids", format!("{:.2} MB", text_bytes_len as f64 / 1_048_576.0)),
        &large_text,
        |b, text| {
            b.iter(|| {
                // Полностью изолируем все входные и мутирующие параметры через black_box,
                // заставляя Out-of-Order движок процессора честно пересчитывать AVX2-маски
                let count = SimdSplitter::split(black_box(text), black_box(&mut ids_buffer), black_box(&byte_fallback));
                black_box(count);
            });
        },
    );

    group.finish();
}

criterion_group!(benches, bench_simd_splitter);
criterion_main!(benches);
