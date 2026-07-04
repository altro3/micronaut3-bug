use super::aggregator::CorpusAggregator;
use crate::tokenizer::dfa::runtime::FlatDfaRuntime;
use crate::tokenizer::factory::types::{AddedToken, BpeModelFields, PreTokenizerEntry, PreTokenizerFields, QwenJsonModel, RegexPattern};
use crate::tokenizer::trainer::{BpeWorker, BuildTrainerHasher};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Error, ErrorKind};
use std::time::Instant;

pub struct TrainerConfig {
    pub batch_size: usize,
    pub initial_table_size: usize,
    pub io_buffer_size: usize,
    pub start_token_id: u32,
}

impl Default for TrainerConfig {
    #[inline(always)]
    fn default() -> Self {
        Self {
            batch_size: 512,
            initial_table_size: 262144,
            io_buffer_size: 2 * 1024 * 1024,
            start_token_id: 256,
        }
    }
}

pub struct BpeTrainer {
    vocab_size: usize,
    config: TrainerConfig,
    dfa_splitter: FlatDfaRuntime,
}

impl BpeTrainer {
    pub fn new(vocab_size: usize, config: TrainerConfig, dfa_splitter: FlatDfaRuntime) -> Self {
        Self {
            vocab_size,
            config,
            dfa_splitter,
        }
    }

    pub fn train(&self, text: &str, output_json_path: &str) -> std::io::Result<()> {
        let timer = Instant::now();
        let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);

        let (global_words, global_counts) = CorpusAggregator::collect_unique_words(text, &self.dfa_splitter, num_threads);

        let mut id_to_bytes = HashMap::with_hasher(BuildTrainerHasher);
        let mut merges: Vec<[String; 2]> = Vec::with_capacity(self.vocab_size);
        let mut vocab_json_output = HashMap::new();

        for b in 0..=255 {
            let b_vec = vec![b];
            id_to_bytes.insert(b as u32, b_vec.clone());
            let qwen_str = Self::bytes_to_qwen_string(&b_vec);
            vocab_json_output.insert(qwen_str, b as u32);
        }

        let w_chunk_size = (global_words.len() + num_threads - 1) / num_threads;
        let mut workers: Vec<Box<BpeWorker>> = Vec::with_capacity(num_threads);
        let dynamic_table_size = (w_chunk_size * 2).max(self.config.initial_table_size).next_power_of_two();

        for i in 0..num_threads {
            let start = (i * w_chunk_size).min(global_words.len());
            let end = ((i + 1) * w_chunk_size).min(global_words.len());
            if start < end {
                workers.push(Box::new(BpeWorker::with_capacity(
                    global_words[start..end].to_vec(),
                    global_counts[start..end].to_vec(),
                    dynamic_table_size,
                )));
            }
        }

        let mut current_id = self.config.start_token_id;
        let b_size = self.config.batch_size;

        println!("[ТРЕНЕР] Начинаю итерационный цикл слияния пар токенов...");

        while id_to_bytes.len() < self.vocab_size {
            let mut global_pair_counts: HashMap<u64, isize, BuildTrainerHasher> = HashMap::with_hasher(BuildTrainerHasher);

            for worker in &workers {
                let limit = worker.table_keys.len();
                let mut taken = 0;
                for idx in 0..limit {
                    let key = worker.table_keys[idx];
                    let stat = worker.table_stats[idx];
                    if key != u64::MAX && stat > 0 {
                        *global_pair_counts.entry(key).or_insert(0) += stat as isize;
                        taken += 1;
                        if taken >= b_size * 2 {
                            break;
                        }
                    }
                }
            }

            let mut pairs_pool: Vec<(u64, isize)> = global_pair_counts.into_iter().collect();
            pairs_pool.sort_unstable_by_key(|&(_, count)| -count);
            if pairs_pool.is_empty() || pairs_pool[0].1 <= 0 {
                break;
            }

            let actual_batch = b_size.min(pairs_pool.len()).min(self.vocab_size - id_to_bytes.len());
            let mut batch_merges = Vec::with_capacity(actual_batch);

            for i in 0..actual_batch {
                let pack = pairs_pool[i].0;
                let (id1, id2) = ((pack >> 32) as u32, pack as u32);
                let mut merged_bytes = id_to_bytes.get(&id1).cloned().unwrap_or_default();
                merged_bytes.extend_from_slice(&id_to_bytes.get(&id2).cloned().unwrap_or_default());

                id_to_bytes.insert(current_id, merged_bytes.clone());

                let qwen_str = Self::bytes_to_qwen_string(&merged_bytes);
                vocab_json_output.insert(qwen_str, current_id);

                merges.push([id1.to_string(), id2.to_string()]);
                batch_merges.push((id1, id2, current_id, pack));
                current_id += 1;
            }

            std::thread::scope(|scope| {
                for worker in &mut workers {
                    let merges_ref = &batch_merges;
                    scope.spawn(move || {
                        for &(id1, id2, new_id, old_pack) in merges_ref {
                            let idx = (old_pack.wrapping_mul(0x517cc1b727220a95) as usize) & worker.mask;
                            worker.table_stats[idx] = 0;
                            let head = worker.table_heads[idx];
                            if head != u32::MAX {
                                let mut curr_w = head as usize;
                                while curr_w != u32::MAX as usize {
                                    let weight = unsafe { *worker.word_counts.get_unchecked(curr_w) } as i64;
                                    worker.merge_tokens_inplace(curr_w, id1, id2, new_id, weight);
                                    curr_w = worker.next_node[curr_w] as usize;
                                }
                            }
                        }
                    });
                }
            });
        }

        println!("[ТРЕНЕР] Слияния завершены. Упаковываю структуры данных в JSON через Serde...");

        let file = File::create(output_json_path)?;
        let writer = BufWriter::with_capacity(self.config.io_buffer_size, file);

        let eos_token_id = current_id;

        let eos_token = AddedToken {
            id: eos_token_id,
            content: "<|endoftext|>".to_string(),
            single_word: false,
            lstrip: false,
            rstrip: false,
            normalized: false,
            special: true,
        };

        let full_model = QwenJsonModel {
            version: "1.0".to_string(),
            added_tokens: Some(vec![eos_token]),
            pre_tokenizer: PreTokenizerFields {
                pretokenizers: vec![PreTokenizerEntry {
                    pattern: Some(RegexPattern {
                        regex: r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]+|\p{L}+|\p{N}{1,3}".to_string(),
                    }),
                }],
            },
            model: BpeModelFields {
                vocab: vocab_json_output,
                merges,
            },
        };

        serde_json::to_writer(writer, &full_model).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;

        println!("[ТРЕНЕР] Обучение успешно завершено! Итоговый словарь сохранен в: {}", output_json_path);
        println!("[ТРЕНЕР] Итоговый размер словаря: {} токенов", eos_token_id + 1);
        println!("[ТРЕНЕР] Полное время работы пайплайна: {:?}", timer.elapsed());
        Ok(())
    }

    fn bytes_to_qwen_string(bytes: &[u8]) -> String {
        let mut result = String::with_capacity(bytes.len() * 4);
        for &b in bytes {
            if (33..=126).contains(&b) && b != b'"' && b != b'\\' {
                result.push(b as char);
            } else {
                result.push(char::from_u32(0x100000 + b as u32).unwrap_or('\u{FFFD}'));
            }
        }
        result
    }
}
