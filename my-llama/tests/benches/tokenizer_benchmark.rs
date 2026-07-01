use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use rustc_hash::FxHashMap;
use my_llama::token::{BpeTokenizer, BpeValue};

fn generate_heavy_tokenizer() -> BpeTokenizer {
    let mut pair_ranks = FxHashMap::default();
    let mut byte_fallback = [0u32; 256];
    for b in 0..=255 { byte_fallback[b] = b as u32; }

    for i in 0..100_000 {
        let id1 = (i % 250) as u64;
        let id2 = ((i + 1) % 250) as u64;
        let pack = (id1 << 32) | id2;
        pair_ranks.insert(pack, BpeValue { rank: i as u32, id: 1000 + i as u32 });
    }

    BpeTokenizer::new(pair_ranks, byte_fallback, 151643)
}

fn bench_tokenizer(c: &mut Criterion) {
    let tokenizer = generate_heavy_tokenizer();

    let base_text = "The quick brown fox jumps over the lazy dog. 1234567890! \n";
    let large_text = base_text.repeat(25_000);

    let mut group = c.benchmark_group("BPE_Max_Speed");
    group.throughput(Throughput::Bytes(large_text.len() as u64));

    group.bench_function("encode_single_thread", |b| {
        b.iter(|| tokenizer.encode(&large_text))
    });

    group.finish();
}

criterion_group!(benches, bench_tokenizer);
criterion_main!(benches);
