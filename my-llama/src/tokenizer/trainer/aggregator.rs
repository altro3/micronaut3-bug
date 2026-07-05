use rayon::prelude::*;
use regex_automata::dfa::dense::DFA;
use regex_automata::dfa::{dense, Automaton};
use regex_automata::util::syntax;
use regex_automata::Input;
use std::collections::HashMap;

pub struct CorpusAggregator {
    dfa: DFA<Vec<u32>>,
    initial_capacity: usize,
}

impl CorpusAggregator {
    pub fn new(cyrillic_regex: &str, initial_capacity: usize) -> Self {
        let syntax_config = syntax::Config::new().utf8(true).multi_line(true);

        let dfa = dense::Builder::new()
            .syntax(syntax_config)
            .build(cyrillic_regex)
            .expect("Ошибка компиляции регулярного автомата BPE с гарантией UTF-8");
        Self { dfa, initial_capacity }
    }

    pub fn collect_unique_words(&self, text_bytes: &[u8], num_threads: usize, local_cap: usize) -> HashMap<Vec<u8>, usize> {
        if text_bytes.is_empty() {
            return HashMap::new();
        }

        let total_len = text_bytes.len();
        let estimated_chunk = (total_len + num_threads - 1) / num_threads;

        let mut boundaries = Vec::with_capacity(num_threads + 1);
        boundaries.push(0);

        for t_idx in 1..num_threads {
            let target_pos = t_idx * estimated_chunk;
            if target_pos >= total_len {
                break;
            }

            let input = Input::new(text_bytes).span(target_pos..total_len);
            let mut pos = target_pos;
            let mut state = self.dfa.start_state_forward(&input).unwrap();

            while pos < total_len {
                state = self.dfa.next_state(state, text_bytes[pos]);
                if self.dfa.is_match_state(state) {
                    pos += 1;
                    break;
                }
                if self.dfa.is_dead_state(state) {
                    pos += 1;
                    break;
                }
                pos += 1;
            }
            boundaries.push(pos.min(total_len));
        }
        boundaries.push(total_len);
        boundaries.dedup();

        (0..boundaries.len() - 1)
            .into_par_iter()
            .map(|t_idx| {
                let start_pos = boundaries[t_idx];
                let end_pos = boundaries[t_idx + 1];

                let mut local_map = HashMap::with_capacity(local_cap);
                let mut pos = start_pos;

                while pos < end_pos {
                    let input = Input::new(text_bytes).span(pos..end_pos);
                    let mut state = self.dfa.start_state_forward(&input).unwrap();
                    let mut match_len = 0;

                    for (i, &b) in text_bytes[pos..end_pos].iter().enumerate() {
                        state = self.dfa.next_state(state, b);
                        if self.dfa.is_match_state(state) {
                            match_len = i + 1;
                        } else if self.dfa.is_dead_state(state) {
                            break;
                        }
                    }

                    if match_len > 0 {
                        let word_bytes = &text_bytes[pos..pos + match_len];
                        *local_map.entry(word_bytes.to_vec()).or_insert(0) += 1;
                        pos += match_len;
                    } else {
                        let fallback_word = vec![text_bytes[pos]];
                        *local_map.entry(fallback_word).or_insert(0) += 1;
                        pos += 1;
                    }
                }
                local_map
            })
            .reduce(
                || HashMap::with_capacity(self.initial_capacity),
                |mut main_map, local_map| {
                    for (word, count) in local_map {
                        *main_map.entry(word).or_insert(0) += count;
                    }
                    main_map
                },
            )
    }
}
