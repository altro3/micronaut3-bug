use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use my_llama::tokenizer::simd_splitter::{SimdSplitter, TokenSpan};
use std::hint::black_box;

fn bench_simd_splitter(c: &mut Criterion) {
    let mut group = c.benchmark_group("SimdSplitter");

    let base_sample = "unbelievable_industrial_grade_artificial_intelligence_tokenizer_performance_test_on_rust ";
    let iterations = 200_000;
    let mut large_text = String::with_capacity(base_sample.len() * iterations);
    for _ in 0..iterations {
        large_text.push_str(base_sample);
    }

    let text_bytes_len = large_text.len();

    group.throughput(Throughput::Bytes(text_bytes_len as u64));

    let mut tokens_buffer = vec![TokenSpan { start: 0, end: 0 }; iterations * 6];

    group.bench_with_input(
        BenchmarkId::new("split_to_spans", format!("{:.1} MB", text_bytes_len as f64 / 1_048_576.0)),
        &large_text,
        |b, text| {
            b.iter(|| {
                let count = SimdSplitter::split(black_box(text), black_box(&mut tokens_buffer));
                black_box(count);
            });
        },
    );

    group.finish();
}

criterion_group!(benches, bench_simd_splitter);
criterion_main!(benches);
