use fxhash::FxHashMap;
use std::time::Instant;

pub struct IsolatedWord {
    pub tokens: Vec<u32>,
    pub weight: i64,
}

pub struct PositionIndex {
    pub pair_counts: FxHashMap<(u32, u32), i64>,
    pub pair_to_words: FxHashMap<(u32, u32), Vec<u32>>,
}

impl PositionIndex {
    pub fn build(unique_words: FxHashMap<Vec<u8>, usize>) -> (Vec<IsolatedWord>, Self) {
        println!("[ИНДЕКСАТОР] Сборка высокоскоростной структуры слов Hugging Face...");
        let start_build = Instant::now();

        let mut pair_counts = FxHashMap::with_capacity_and_hasher(unique_words.len() * 2, Default::default());
        let mut pair_to_words = FxHashMap::with_capacity_and_hasher(unique_words.len() * 2, Default::default());
        let mut words = Vec::with_capacity(unique_words.len());

        for (bytes, count) in unique_words {
            if bytes.is_empty() {
                continue;
            }

            let word_id = words.len() as u32;
            let weight = count as i64;
            let tokens: Vec<u32> = bytes.into_iter().map(|b| b as u32).collect();

            let mut last_pair = (u32::MAX, u32::MAX);
            for window in tokens.windows(2) {
                let pair = (window[0], window[1]);
                *pair_counts.entry(pair).or_insert(0) += weight;

                if pair != last_pair {
                    pair_to_words.entry(pair).or_insert_with(Vec::new).push(word_id);
                    last_pair = pair;
                }
            }

            words.push(IsolatedWord { tokens, weight });
        }

        words.sort_unstable_by_key(|w| w.tokens.len());

        println!("  ├── Уникальных отсортированных слов в базе: {}", words.len());
        println!("  ├── Уникальных пар на старте: {}", pair_counts.len());
        println!("  └── Индексация завершена за: {:?}", start_build.elapsed());

        (words, Self { pair_counts, pair_to_words })
    }
}
