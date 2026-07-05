use crate::tokenizer::trainer::flat_corpus::FlatCorpus;
use std::collections::HashMap;

pub struct PositionMatch {
    pub word_idx: usize,
    pub token_idx: usize,
    pub id1: u32,
    pub id2: u32,
    pub new_id: u32,
}

pub struct InvertedIndex {
    pub pair_locations: HashMap<(u32, u32), Vec<(u32, u32)>>,
}

impl InvertedIndex {
    pub fn new() -> Self {
        Self {
            pair_locations: HashMap::with_capacity(524288),
        }
    }

    pub fn rebuild(&mut self, corpus: &FlatCorpus) {
        self.pair_locations.clear();

        for (w_idx, word) in corpus.words.iter().enumerate() {
            if word.tokens.len() < 2 { continue; }
            for t_idx in 0..word.tokens.len() - 1 {
                let pair = (word.tokens[t_idx], word.tokens[t_idx + 1]);
                self.pair_locations
                    .entry(pair)
                    .or_insert_with(|| Vec::with_capacity(4))
                    .push((w_idx as u32, t_idx as u32));
            }
        }
    }
}
