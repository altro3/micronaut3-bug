use rayon::prelude::*;
use regex_automata::dfa::dense::DFA;
use regex_automata::dfa::Automaton;
use regex_automata::Input;
use std::collections::HashMap;

pub struct CorpusAggregator {
    dfa: DFA<Vec<u32>>,
    initial_capacity: usize,
}

impl CorpusAggregator {
    pub fn new(cyrillic_regex: &str, initial_capacity: usize) -> Self {
        let dfa = DFA::new(cyrillic_regex).expect("Ошибка компиляции регулярного автомата BPE");
        Self { dfa, initial_capacity }
    }

    pub fn collect_unique_words(&self, text_bytes: &[u8], num_threads: usize, local_cap: usize) -> HashMap<Vec<u8>, usize> {
        let chunk_size = (text_bytes.len() + num_threads - 1) / num_threads;
        let chunks: Vec<&[u8]> = text_bytes.chunks(chunk_size).collect();

        chunks
            .into_par_iter()
            .map(|chunk| {
                let mut local_map = HashMap::with_capacity(local_cap);
                let mut pos = 0;

                while pos < chunk.len() {
                    let search_input = Input::new(&chunk[pos..]);
                    let mut state = self.dfa.start_state_forward(&search_input).unwrap();
                    let mut match_len = 0;

                    for (i, &b) in chunk[pos..].iter().enumerate() {
                        state = self.dfa.next_state(state, b);
                        if self.dfa.is_match_state(state) {
                            match_len = i + 1;
                        } else if self.dfa.is_dead_state(state) {
                            break;
                        }
                    }

                    if match_len > 0 {
                        let word_bytes = &chunk[pos..pos + match_len];
                        *local_map.entry(word_bytes.to_vec()).or_insert(0) += 1;
                        pos += match_len;
                    } else {
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
