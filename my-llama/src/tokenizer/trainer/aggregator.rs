use rayon::prelude::*;
use regex::Regex;
use std::collections::HashMap;

pub struct CorpusAggregator {
    regex: Regex,
    initial_capacity: usize,
}

impl CorpusAggregator {
    pub fn new(cyrillic_regex: &str, initial_capacity: usize) -> Self {
        let regex = Regex::new(cyrillic_regex).expect("Ошибка компиляции регулярного выражения BPE");
        Self { regex, initial_capacity }
    }

    pub fn collect_unique_words(&self, text_bytes: &[u8], _num_threads: usize, local_cap: usize) -> HashMap<Vec<u8>, usize> {
        if text_bytes.is_empty() {
            return HashMap::new();
        }

        let text_str = std::str::from_utf8(text_bytes).expect("Корпус содержит невалидный UTF-8");

        let mut line_boundaries = Vec::new();
        line_boundaries.push(0);
        for (idx, &b) in text_bytes.iter().enumerate() {
            if b == b'\n' {
                line_boundaries.push(idx + 1);
            }
        }
        line_boundaries.push(text_bytes.len());

        (0..line_boundaries.len() - 1)
            .into_par_iter()
            .map(|chunk_idx| {
                let mut local_map = HashMap::with_capacity(local_cap);

                let start_pos = line_boundaries[chunk_idx];
                let end_pos = line_boundaries[chunk_idx + 1];

                let line_str = &text_str[start_pos..end_pos];
                if line_str.is_empty() {
                    return local_map;
                }

                for mat in self.regex.find_iter(line_str) {
                    let word_str = mat.as_str();
                    if word_str.is_empty() {
                        continue;
                    }
                    let word_bytes = word_str.as_bytes().to_vec();

                    *local_map.entry(word_bytes).or_insert(0) += 1;
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
