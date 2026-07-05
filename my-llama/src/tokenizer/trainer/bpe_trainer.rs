use rayon::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

use crate::tokenizer::trainer::aggregator::CorpusAggregator;
use crate::tokenizer::trainer::config::{MergeCommand, TrainerConfig};
use crate::tokenizer::trainer::exporter::VocabularyExporter;
use crate::tokenizer::trainer::flat_corpus::{FlatCorpus, WordEntry};
use crate::tokenizer::trainer::utils::TrainerUtils;
use crate::tokenizer::trainer::worker::ThreadDeltaWorker;

pub struct BpeTrainer {
    vocab_size: usize,
    config: TrainerConfig,
    cyrillic_regex: &'static str,
}

impl BpeTrainer {
    pub fn new(vocab_size: usize, config: TrainerConfig) -> Self {
        Self {
            vocab_size,
            config,
            cyrillic_regex: r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}+|(?:\s)[\r\n]*|\s+[\r\n]*|[\r\n]+",
        }
    }

    pub fn train(&self, text_content: &str, output_json_path: &str) -> std::io::Result<()> {
        println!("[HPC ТРЕНЕР] Шаг 1: Агрегация корпуса через регулярный автомат...");
        let global_timer = Instant::now();

        let aggregator = CorpusAggregator::new(self.cyrillic_regex, self.config.initial_table_size);
        let unique_words = aggregator.collect_unique_words(
            text_content.as_bytes(),
            self.config.num_threads,
            self.config.local_map_capacity,
        );

        println!("[HPC ТРЕНЕР] Шаг 2: Построение изолированных доменов слов...");
        let (mut corpus, global_pairs) = FlatCorpus::build(unique_words);

        let mut current_id = self.config.start_token_id;
        let mut iteration = 0;

        let mut id_to_bytes: Vec<Vec<u8>> = (0..256).map(|b| vec![b as u8]).collect();
        id_to_bytes.reserve(self.vocab_size);

        let mut merges: Vec<[String; 2]> = Vec::with_capacity(self.vocab_size);
        let mut vocab_json_output = HashMap::with_capacity(self.vocab_size);
        for b in 0..256 {
            let b_vec = vec![b as u8];
            vocab_json_output.insert(TrainerUtils::bytes_to_qwen_string(&b_vec), b as u32);
        }

        let mut global_pair_frequencies = global_pairs;

        println!("[HPC ТРЕНЕР] Шаг 3: Разделение корпуса на изолированные потоки...");
        let chunk_size = (corpus.words.len() + self.config.num_threads - 1) / self.config.num_threads;
        let mut chunks: Vec<Vec<WordEntry>> = (0..self.config.num_threads)
            .map(|_| Vec::with_capacity(chunk_size))
            .collect();

        let mut words_iter = corpus.words.into_iter();
        for t_idx in 0..self.config.num_threads {
            for _ in 0..chunk_size {
                if let Some(word) = words_iter.next() {
                    chunks[t_idx].push(word);
                } else {
                    break;
                }
            }
        }

        println!("[HPC ТРЕНЕР] Запуск Chunk-Parallel BPE на {} потоках...", self.config.num_threads);

        while (current_id as usize) < self.vocab_size {
            let iter_timer = Instant::now();
            iteration += 1;

            let mut flat_pairs: Vec<((u32, u32), i64)> = global_pair_frequencies
                .iter()
                .filter(|&(_, &c)| c > 0)
                .map(|(&p, &c)| (p, c))
                .collect();

            if flat_pairs.is_empty() { break; }

            let total_unique_pairs_in_corpus = flat_pairs.len();

            // Вырезаем батч строго по лимиту (например, 256 пар)
            let b_size = self.config.batch_size.min(flat_pairs.len());
            flat_pairs.select_nth_unstable_by_key(b_size, |&(_, c)| -c);
            flat_pairs.truncate(b_size);

            // Сортируем по убыванию частоты — это СТРОГО определяет приоритет мёржа!
            flat_pairs.sort_unstable_by_key(|&(_, c)| -c);

            let mut targets = Vec::with_capacity(b_size);

            // Лог топ-пары
            let mut top_pair_str = String::from("None");
            if let Some(&(pair, freq)) = flat_pairs.first() {
                if (pair.0 as usize) < id_to_bytes.len() && (pair.1 as usize) < id_to_bytes.len() {
                    let mut b_res = id_to_bytes[pair.0 as usize].clone();
                    b_res.extend_from_slice(&id_to_bytes[pair.1 as usize]);
                    let clean_text = String::from_utf8_lossy(&b_res).into_owned();
                    top_pair_str = format!("'{}' (freq: {})", clean_text.escape_debug(), freq);
                }
            }

            // Набиваем батч ВСЕМИ топовыми парами без каких-либо HashSet фильтраций!
            for (pair, _) in flat_pairs {
                targets.push((pair, current_id));

                let mut merged_bytes = id_to_bytes[pair.0 as usize].clone();
                merged_bytes.extend_from_slice(&id_to_bytes[pair.1 as usize]);
                let str_a = TrainerUtils::bytes_to_qwen_string(&id_to_bytes[pair.0 as usize]);
                let str_b = TrainerUtils::bytes_to_qwen_string(&id_to_bytes[pair.1 as usize]);
                vocab_json_output.insert(TrainerUtils::bytes_to_qwen_string(&merged_bytes), current_id);
                merges.push([str_a, str_b]);
                id_to_bytes.push(merged_bytes);

                // ИСПРАВЛЕНИЕ: Обнуляем частоту этой пары в правильной мапе
                global_pair_frequencies.insert(pair, 0);

                current_id += 1;
                if (current_id as usize) >= self.vocab_size { break; }
            }

            if targets.is_empty() { continue; }

            let targets_ref = &targets;

            // Тяжелый, честный параллельный проход по словам
            let local_deltas: Vec<HashMap<(u32, u32), i64>> = chunks
                .par_iter_mut()
                .map(|word_chunk| {
                    let mut chunk_delta = HashMap::with_capacity(2048);
                    for word in word_chunk {
                        ThreadDeltaWorker::process_word_locally(word, targets_ref, &mut chunk_delta);
                    }
                    chunk_delta
                })
                .collect();

            // Консолидация изменений частот
            for delta_map in local_deltas {
                for (pair, delta) in delta_map {
                    let count = global_pair_frequencies.entry(pair).or_insert(0);
                    *count += delta;
                    if *count < 0 { *count = 0; }
                }
            }

            let progress = (current_id as f64 / self.vocab_size as f64) * 100.0;

            println!(
                "[BPE ЭТАП] Шаг: {:<4} | Токенов: {}/{} ({:.2}%) | Живых пар в базе: {:<6} | Мёржей в батче: {:<3} | Топ-1 пара: {:<30} | Время: {:?}",
                iteration, current_id, self.vocab_size, progress, total_unique_pairs_in_corpus, targets_ref.len(), top_pair_str, iter_timer.elapsed()
            );
        }

        println!("[ЭКСПОРТ] Запись JSON структуры на диск...");
        VocabularyExporter::export_qwen_json(output_json_path, &self.config, self.cyrillic_regex, vocab_json_output, merges, current_id)?;
        println!("[УСПЕХ] Весь процесс токенизации завершен за: {:?}", global_timer.elapsed());
        Ok(())
    }
}
