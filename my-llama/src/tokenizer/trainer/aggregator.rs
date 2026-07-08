use rayon::prelude::*;
use regex::bytes::Regex;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

pub struct CorpusAggregator {
    regex: Regex,
    initial_capacity: usize,
}

impl CorpusAggregator {
    pub fn new(regex_str: &str, initial_capacity: usize) -> Self {
        let regex = Regex::new(regex_str).expect("Ошибка компиляции регулярного выражения BPE");
        Self { regex, initial_capacity }
    }

    pub fn collect_unique_words(&self, text_bytes: &[u8], _num_threads: usize, local_cap: usize) -> HashMap<Vec<u8>, usize> {
        if text_bytes.is_empty() {
            return HashMap::new();
        }

        println!("[АГРЕГАТОР] Тотальный многопоточный анализ СЫРЫХ БАЙТ (100% без строк)...");
        let start_agg = Instant::now();

        let broken_logs = Mutex::new(Vec::new());

        let mut line_boundaries = Vec::new();
        line_boundaries.push(0);
        for (idx, &b) in text_bytes.iter().enumerate() {
            if b == b'\n' {
                line_boundaries.push(idx + 1);
            }
        }
        line_boundaries.push(text_bytes.len());

        let total_lines = line_boundaries.len() - 1;

        let merged_map = (0..total_lines)
            .into_par_iter()
            .map(|chunk_idx| {
                let mut local_map = HashMap::with_capacity(local_cap);
                let start_pos = line_boundaries[chunk_idx];
                let end_pos = line_boundaries[chunk_idx + 1];

                let line_raw_bytes = &text_bytes[start_pos..end_pos];
                if line_raw_bytes.is_empty() {
                    return local_map;
                }

                for mat in self.regex.find_iter(line_raw_bytes) {
                    let word_bytes = mat.as_bytes();
                    if word_bytes.is_empty() {
                        continue;
                    }

                    if let Err(utf8_err) = std::str::from_utf8(word_bytes) {
                        let mut logs = broken_logs.lock().unwrap();

                        let ctx_start = if mat.start() > 30 { mat.start() - 30 } else { 0 };
                        let ctx_end = if mat.end() + 30 < line_raw_bytes.len() { mat.end() + 30 } else { line_raw_bytes.len() };
                        let context_bytes = &line_raw_bytes[ctx_start..ctx_end];

                        let visible_context = String::from_utf8_lossy(context_bytes).into_owned();

                        logs.push(format!(
                            "Строка #{:<6} | Смещение: {}-{} | Ошибка: {:?} | Сырые байты токена: {:?} | Контекст: \"... {} ...\"",
                            chunk_idx,
                            mat.start(),
                            mat.end(),
                            utf8_err,
                            word_bytes,
                            visible_context.trim().escape_debug()
                        ));
                    }

                    let raw_bytes = word_bytes.to_vec();
                    *local_map.entry(raw_bytes).or_insert(0) += 1;
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
            );

        println!("  ├── Агрегация завершена за: {:?}", start_agg.elapsed());
        println!("  └── Найдено unique слов: {}", merged_map.len());

        let final_logs = broken_logs.into_inner().unwrap();
        if !final_logs.is_empty() {
            println!(
                "\n[КРИТИЧЕСКИЙ ДАМП АГРЕГАТОРА] Поймали реальные ломаные байты из corpus.txt: {}",
                final_logs.len()
            );
            println!("------------------------------------------------------------------------------------------------------------------------");
            for log in final_logs {
                println!("  [БАГ] {}", log);
            }
            println!("------------------------------------------------------------------------------------------------------------------------\n");
        } else {
            println!("  └── [ОК] Регулярка не зацепила ни одного ломаного UTF-8 куска.");
        }

        merged_map
    }
}
