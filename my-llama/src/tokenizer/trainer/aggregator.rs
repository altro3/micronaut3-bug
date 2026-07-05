use rayon::prelude::*;
use regex::Regex;
use std::collections::HashMap;
use crate::tokenizer::trainer::utils::TrainerUtils;

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

        // Превращаем байты корпуса в валидную строку через lossy, чтобы защититься от грязи в файле
        let text_str = String::from_utf8_lossy(text_bytes);

        // Нарезаем границы строк по символу '\n'
        let mut line_boundaries = Vec::new();
        line_boundaries.push(0);
        for (idx, b) in text_str.bytes().enumerate() {
            if b == b'\n' {
                line_boundaries.push(idx + 1);
            }
        }
        line_boundaries.push(text_str.len());

        (0..line_boundaries.len() - 1)
            .into_par_iter()
            .map(|chunk_idx| {
                let mut local_map = HashMap::with_capacity(local_cap);

                let start_pos = line_boundaries[chunk_idx];
                let end_pos = line_boundaries[chunk_idx + 1];

                let line_str = &text_str[start_pos..end_pos];
                if line_str.is_empty() { return local_map; }

                let qwen_line_str = TrainerUtils::bytes_to_qwen_string(line_str.as_bytes());

                for mat in self.regex.find_iter(&qwen_line_str) {
                    let word_qwen_str = mat.as_str();
                    if word_qwen_str.is_empty() { continue; }
                    let original_raw_bytes = TrainerUtils::qwen_string_to_bytes(word_qwen_str);

                    *local_map.entry(original_raw_bytes).or_insert(0) += 1;
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
