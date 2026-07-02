use my_llama::tokenizer::{BpeTokenizer, BpeValue};
use rustc_hash::FxHashMap;

fn create_test_tokenizer() -> BpeTokenizer {
    let mut pair_ranks = FxHashMap::default();
    let mut byte_fallback = [0u32; 256];

    for b in 0..=255 {
        byte_fallback[b] = b as u32;
    }

    let pack_ab = ((97u64) << 32) | (98u64);
    pair_ranks.insert(pack_ab, BpeValue { rank: 0, id: 500 });

    let pack_abc = ((500u64) << 32) | (99u64);
    pair_ranks.insert(pack_abc, BpeValue { rank: 1, id: 501 });

    BpeTokenizer::new(pair_ranks, byte_fallback, 151643)
}

#[test]
fn test_short_chunk_merging() {
    let tokenizer = create_test_tokenizer();
    let tokens = tokenizer.encode("abc");
    assert_eq!(tokens, vec![501]);
}

#[test]
fn test_long_chunk_merging() {
    let tokenizer = create_test_tokenizer();
    let text = "abc_abc_abc_abc_abc_abc";
    let tokens = tokenizer.encode(text);
    assert!(tokens.contains(&501));
}

#[test]
fn test_empty_string() {
    let tokenizer = create_test_tokenizer();
    let tokens = tokenizer.encode("");
    assert!(tokens.is_empty());
}

#[test]
fn test_parallel_encoding() {
    let tokenizer = create_test_tokenizer();
    let input = vec!["abc".to_string(), "abc_abc".to_string(), "".to_string()];

    let results = tokenizer.encode_parallel(&input);

    assert_eq!(results.len(), 3);
    assert_eq!(results[0], vec![501]);
    assert_eq!(results[1], vec![501, 95, 501]);
    assert!(results[2].is_empty());
}
