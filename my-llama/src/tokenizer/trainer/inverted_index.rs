use crate::tokenizer::trainer::flat_corpus::FlatCorpus;
use rayon::prelude::*;
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

    pub fn rebuild(&mut self, corpus: &FlatCorpus, num_threads: usize) {
        let total_tokens = corpus.tokens.len();
        if total_tokens < 2 {
            return;
        }

        self.next_pos.fill(-1);

        let chunk_size = (total_tokens + num_threads - 1) / num_threads;

        let thread_indices: Vec<HashMap<(u32, u32), i32>> = (0..num_threads)
            .into_par_iter()
            .map(|t_idx| {
                let start = (t_idx * chunk_size).min(total_tokens);
                let mut end = ((t_idx + 1) * chunk_size).min(total_tokens);

                if t_idx < num_threads - 1 && end < total_tokens {
                    end += 1;
                }

                let mut local_heads: HashMap<(u32, u32), i32> = HashMap::with_capacity(32768);
                if start >= end {
                    return local_heads;
                }

                let mut curr = start;
                while curr < end - 1 {
                    let next_node = corpus.next[curr];
                    if next_node != -1 && (next_node as usize) < end {
                        let pair = (corpus.tokens[curr], corpus.tokens[next_node as usize]);

                        let next_pos_ptr = self.next_pos.as_ptr() as *mut i32;
                        unsafe {
                            if let Some(&head) = local_heads.get(&pair) {
                                *next_pos_ptr.add(curr) = head;
                            }
                        }
                        local_heads.insert(pair, curr as i32);
                    }
                    curr += 1;
                }
                local_heads
            })
            .collect();

        self.head_pairs.clear();
        for local_heads in thread_indices {
            for (pair, local_head) in local_heads {
                if let Some(&global_head) = self.head_pairs.get(&pair) {
                    let mut tail = local_head as usize;
                    while self.next_pos[tail] != -1 {
                        tail = self.next_pos[tail] as usize;
                    }
                    self.next_pos[tail] = global_head;
                }
                self.head_pairs.insert(pair, local_head);
            }
        }
    }
}
