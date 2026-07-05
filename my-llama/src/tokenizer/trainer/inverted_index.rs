use crate::tokenizer::trainer::flat_corpus::FlatCorpus;
use std::collections::HashMap;

pub struct InvertedIndex {
    pub head_pairs: HashMap<(u32, u32), i32>,
    pub next_pos: Vec<i32>,
}

impl InvertedIndex {
    pub fn new(total_tokens: usize) -> Self {
        Self {
            head_pairs: HashMap::with_capacity(524288),
            next_pos: vec![-1; total_tokens],
        }
    }

    pub fn rebuild(&mut self, corpus: &FlatCorpus) {
        self.head_pairs.clear();
        self.next_pos.fill(-1);

        for &start in &corpus.word_starts {
            let mut curr = start as i32;
            while curr != -1 {
                let next_node = corpus.next[curr as usize];
                if next_node != -1 {
                    let pair = (corpus.tokens[curr as usize], corpus.tokens[next_node as usize]);
                    if let Some(&head) = self.head_pairs.get(&pair) {
                        self.next_pos[curr as usize] = head;
                    }
                    self.head_pairs.insert(pair, curr);
                }
                curr = next_node;
            }
        }
    }
}
