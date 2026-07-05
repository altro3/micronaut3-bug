use crate::tokenizer::trainer::flat_corpus::FlatCorpus;
use std::collections::{HashMap, HashSet};

pub struct InvertedIndex {
    pub pair_to_words: HashMap<(u32, u32), HashSet<usize>>,
}

impl InvertedIndex {
    pub fn build(corpus: &FlatCorpus) -> Self {
        let mut pair_to_words: HashMap<(u32, u32), HashSet<usize>> = HashMap::with_capacity(524288);

        for (w_idx, word) in corpus.words.iter().enumerate() {
            let mut curr_node_idx = word.head;
            while curr_node_idx != -1 {
                let node = corpus.nodes[curr_node_idx as usize];

                if node.next != -1 {
                    let next_node = corpus.nodes[node.next as usize];
                    let pair = (node.id, next_node.id);

                    pair_to_words.entry(pair).or_insert_with(HashSet::new).insert(w_idx);

                    curr_node_idx = node.next;
                } else {
                    break;
                }
            }
        }

        Self { pair_to_words }
    }

    #[inline(always)]
    pub fn update_pair(&mut self, pair: (u32, u32), word_idx: usize, exists: bool) {
        if exists {
            self.pair_to_words.entry(pair).or_insert_with(HashSet::new).insert(word_idx);
        } else if let Some(words) = self.pair_to_words.get_mut(&pair) {
            words.remove(&word_idx);
        }
    }
}
