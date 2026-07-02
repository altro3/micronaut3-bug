use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use my_llama::tokenizer::{BpeTokenizer, BpeValue};
use rustc_hash::FxHashMap;

fn generate_heavy_tokenizer() -> BpeTokenizer {
    let mut pair_ranks = FxHashMap::default();
    let mut byte_fallback = [0u32; 256];
    for b in 0..=255 {
        byte_fallback[b] = b as u32;
    }

    for i in 0..100_000 {
        let id1 = (i % 250) as u64;
        let id2 = ((i + 1) % 250) as u64;
        let pack = (id1 << 32) | id2;
        pair_ranks.insert(
            pack,
            BpeValue {
                rank: i as u32,
                id: 1000 + i as u32,
            },
        );
    }

    BpeTokenizer::new(pair_ranks, byte_fallback, 151643)
}

fn bench_tokenizer(c: &mut Criterion) {
    let tokenizer = generate_heavy_tokenizer();

    let base_text = "The quick brown fox jumps over the lazy dog. 1234567890! \n";
    let large_text = base_text.repeat(25_000);

    let mut group = c.benchmark_group("BPE_Max_Speed");
    group.throughput(Throughput::Bytes(large_text.len() as u64));
    group.bench_function("encode_single_thread", |b| b.iter(|| tokenizer.encode(&large_text)));
    group.finish();

    let batch_texts: Vec<String> = (0..100).map(|_| base_text.repeat(250)).collect();
    let total_batch_bytes: usize = batch_texts.iter().map(|s| s.len()).sum();

    let mut group_parallel = c.benchmark_group("BPE_Parallel");
    group_parallel.throughput(Throughput::Bytes(total_batch_bytes as u64));

    group_parallel.bench_function("encode_multi_thread", |b| b.iter(|| tokenizer.encode_parallel(&batch_texts)));
    group_parallel.finish();
}

criterion_group!(benches, bench_tokenizer);
criterion_main!(benches);
