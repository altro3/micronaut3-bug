use rayon::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

use crate::tokenizer::trainer::aggregator::CorpusAggregator;
use crate::tokenizer::trainer::config::{PositionMatch, TrainerConfig};
use crate::tokenizer::trainer::exporter::VocabularyExporter;
use crate::tokenizer::trainer::flat_corpus::FlatCorpus;
use crate::tokenizer::trainer::inverted_index::InvertedIndex;
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
            cyrillic_regex: r" ?\p{L}+|\p{L}+| ?\p{N}+|[^\s\p{L}\p{N}]+|\s*[\r\n]+|\s+",
        }
    }

    pub fn train(&self, text_content: &str, output_json_path: &str) -> std::io::Result<()> {
        println!("[HPC ТРЕНЕР] Шаг 1: Запуск параллельного сбора корпуса на 16 ядрах...");
        let global_timer = Instant::now();

        let aggregator = CorpusAggregator::new(self.cyrillic_regex, self.config.initial_table_size);
        let unique_words = aggregator.collect_unique_words(text_content.as_bytes(), self.config.num_threads, self.config.local_map_capacity);

        println!(
            "[HPC ТРЕНЕР] Собрано уникальных слов: {}. Индексация плоского корпуса...",
            unique_words.len()
        );
        let (corpus, mut pair_stats) = FlatCorpus::build(unique_words);

        println!(
            "[HPC ТРЕНЕР] Общий пул токенов: {}. Сборка инвертированного индекса...",
            corpus.tokens.len()
        );
        let mut index = InvertedIndex::new(corpus.tokens.len());
        index.rebuild(&corpus);

        let mut current_id = self.config.start_token_id;
        let mut iteration_count = 0;

        let mut id_to_bytes: Vec<Vec<u8>> = (0..256).map(|b| vec![b as u8]).collect();
        id_to_bytes.reserve(self.vocab_size);

        let mut merges: Vec<[String; 2]> = Vec::with_capacity(self.vocab_size);
        let mut vocab_json_output = HashMap::with_capacity(self.vocab_size);
        for b in 0..256 {
            let b_vec = vec![b as u8];
            vocab_json_output.insert(TrainerUtils::bytes_to_qwen_string(&b_vec), b as u32);
        }

        let mut local_deltas: Vec<HashMap<(u32, u32), i64>> = (0..self.config.num_threads)
            .map(|_| HashMap::with_capacity(self.config.delta_map_capacity))
            .collect();

        println!(
            "[HPC ТРЕНЕР] Запуск пакетного инкрементального сжатия. Целевой размер: {}",
            self.vocab_size
        );

        while (current_id as usize) < self.vocab_size {
            let iter_timer = Instant::now();
            iteration_count += 1;

            let mut flat_pairs: Vec<((u32, u32), i64)> = pair_stats.iter().filter(|&(_, &c)| c > 0).map(|(&p, &c)| (p, c)).collect();
            if flat_pairs.is_empty() {
                break;
            }

            let b_size = self.config.batch_size.min(flat_pairs.len());
            flat_pairs.select_nth_unstable_by_key(b_size, |&(_, c)| -c);
            flat_pairs.truncate(b_size);

            let mut positions_to_process = Vec::with_capacity(self.config.position_buffer_capacity);
            let mut actual_merge_count = 0;

            for (pair, _) in flat_pairs {
                if let Some(&head) = index.head_pairs.get(&pair) {
                    let mut curr = head;
                    while curr != -1 {
                        positions_to_process.push(PositionMatch {
                            pos: curr as usize,
                            id1: pair.0,
                            id2: pair.1,
                            new_id: current_id,
                        });
                        curr = index.next_pos[curr as usize];
                    }
                }

                let mut merged_bytes = id_to_bytes[pair.0 as usize].clone();
                merged_bytes.extend_from_slice(&id_to_bytes[pair.1 as usize]);

                let str_a = TrainerUtils::bytes_to_qwen_string(&id_to_bytes[pair.0 as usize]);
                let str_b = TrainerUtils::bytes_to_qwen_string(&id_to_bytes[pair.1 as usize]);
                vocab_json_output.insert(TrainerUtils::bytes_to_qwen_string(&merged_bytes), current_id);
                merges.push([str_a, str_b]);
                id_to_bytes.push(merged_bytes);

                pair_stats.insert(pair, 0);
                index.head_pairs.remove(&pair);
                current_id += 1;
                actual_merge_count += 1;

                if (current_id as usize) >= self.vocab_size {
                    break;
                }
            }

            if positions_to_process.is_empty() {
                continue;
            }
            positions_to_process.sort_unstable_by_key(|m| m.pos);

            for d in &mut local_deltas {
                d.clear();
            }
            let chunks: Vec<&[PositionMatch]> = positions_to_process
                .chunks((positions_to_process.len() + self.config.num_threads - 1) / self.config.num_threads)
                .collect();

            let chunks_len = chunks.len();
            chunks
                .into_par_iter()
                .zip(&mut local_deltas[0..chunks_len])
                .for_each(|(p_chunk, l_delta)| {
                    ThreadDeltaWorker::process_chunk(p_chunk, &corpus, l_delta);
                });

            for l_delta in &local_deltas {
                for (&pair, &delta) in l_delta {
                    *pair_stats.entry(pair).or_insert(0) += delta;
                }
            }

            let mut index_rebuilt = false;
            if iteration_count % self.config.index_rebuild_interval == 0 {
                index.rebuild(&corpus);
                index_rebuilt = true;
            }

            let progress = (current_id as f64 / self.vocab_size as f64) * 100.0;
            println!(
                "[ПАКЕТНЫЙ ШАГ] Итерация: {:<4} | Токенов: {}/{} ({:.2}%) | Парей: {:<3} | Мутаций: {:<6} | Время: {:?} {}",
                iteration_count,
                current_id,
                self.vocab_size,
                progress,
                actual_merge_count,
                positions_to_process.len(),
                iter_timer.elapsed(),
                if index_rebuilt {
                    "[Индекс дефрагментирован]"
                } else {
                    ""
                }
            );
        }

        println!("[ПРЕД-ЭКСПОРТ] Сжатие завершено. Запись JSON модели...");
        VocabularyExporter::export_qwen_json(output_json_path, &self.config, self.cyrillic_regex, vocab_json_output, merges, current_id)?;

        println!("[УСПЕХ] Весь тренировочный пайплайн отработал за: {:?}", global_timer.elapsed());
        Ok(())
    }
}
