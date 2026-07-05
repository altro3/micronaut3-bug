use fxhash::FxHasher;
use std::hash::BuildHasherDefault;

type FxHashMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<FxHasher>>;
type FxHashSet<T> = std::collections::HashSet<T, BuildHasherDefault<FxHasher>>;

pub struct IsolatedWord {
    pub tokens: Vec<u32>,
    pub weight: i64,
}

pub struct PositionIndex {
    pub pair_counts: FxHashMap<(u32, u32), i64>,
    pub pair_to_words: FxHashMap<(u32, u32), FxHashSet<u32>>,
}

impl PositionIndex {
    pub fn build(unique_words: FxHashMap<Vec<u8>, usize>) -> (Vec<IsolatedWord>, Self) {
        let mut words = Vec::with_capacity(unique_words.len());
        let mut pair_counts = FxHashMap::with_capacity_and_hasher(unique_words.len() * 2, Default::default());
        let mut pair_to_words: FxHashMap<(u32, u32), FxHashSet<u32>> =
            FxHashMap::with_capacity_and_hasher(unique_words.len() * 2, Default::default());

        for (bytes, count) in unique_words {
            if bytes.is_empty() {
                continue;
            }

            let word_id = words.len() as u32;
            let weight = count as i64;
            let tokens: Vec<u32> = bytes.into_iter().map(|b| b as u32).collect();

            for window in tokens.windows(2) {
                let pair = (window[0], window[1]);
                *pair_counts.entry(pair).or_insert(0) += weight;
                pair_to_words.entry(pair).or_insert_with(FxHashSet::default).insert(word_id);
            }

            words.push(IsolatedWord { tokens, weight });
        }

        (words, Self { pair_counts, pair_to_words })
    }
}
