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

        let cyrillic_regex = r" ?\p{L}+|\p{L}+| ?\p{N}+|[^\s\p{L}\p{N}]+|\s*[\r\n]+|\s+";

        // Вызываем полностью автономный агрегатор без сторонних либ
        let (global_words, global_counts) = CorpusAggregator::collect_unique_words(text, cyrillic_regex, num_threads);

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

        println!("[ТРЕНЕР] Начинаю высокопроизводительный пакетный цикл слияния пар...");

        while id_to_bytes.len() < self.vocab_size {
            let mut global_pair_counts: HashMap<u64, isize, BuildTrainerHasher> = HashMap::with_hasher(BuildTrainerHasher);

            for worker in &workers {
                let limit = worker.table_keys.len();
                for idx in 0..limit {
                    let key = worker.table_keys[idx];
                    let stat = worker.table_stats[idx];
                    if key != u64::MAX && stat > 0 {
                        *global_pair_counts.entry(key).or_insert(0) += stat as isize;
                    }
                }
            }

            let mut pairs_pool: Vec<(u64, isize)> = global_pair_counts.into_iter().collect();
            pairs_pool.sort_unstable_by_key(|&(_, count)| -count);

            println!(
                "[ОТЛАДКА BPE] Всего уникальных пар в пуле: {}. Частота топовой пары: {:?}",
                pairs_pool.len(),
                pairs_pool.first().map(|p| p.1)
            );

            if pairs_pool.is_empty() || pairs_pool[0].1 <= 0 {
                println!("[ТРЕНЕР] Больше нет доступных пар с положительной частотой. Остановка.");
                break;
            }

            let mut batch_merges = Vec::with_capacity(b_size);
            let mut seen_tokens = std::collections::HashSet::with_capacity(b_size * 2);

            for (pack, count) in pairs_pool {
                if batch_merges.len() >= b_size || id_to_bytes.len() + batch_merges.len() >= self.vocab_size {
                    break;
                }
                if count <= 0 {
                    break;
                }

                let id1 = (pack >> 32) as u32;
                let id2 = pack as u32;

                if seen_tokens.contains(&id1) || seen_tokens.contains(&id2) {
                    continue;
                }

                seen_tokens.insert(id1);
                seen_tokens.insert(id2);

                batch_merges.push((id1, id2, pack));
            }

            if batch_merges.is_empty() {
                break;
            }

            let mut final_batch_merges = Vec::with_capacity(batch_merges.len());
            for (id1, id2, old_pack) in batch_merges {
                let mut merged_bytes = id_to_bytes.get(&id1).cloned().unwrap_or_default();
                merged_bytes.extend_from_slice(&id_to_bytes.get(&id2).cloned().unwrap_or_default());

                id_to_bytes.insert(current_id, merged_bytes.clone());

                let qwen_str = Self::bytes_to_qwen_string(&merged_bytes);
                vocab_json_output.insert(qwen_str, current_id);

                merges.push([id1.to_string(), id2.to_string()]);
                final_batch_merges.push((id1, id2, current_id, old_pack));
                current_id += 1;
            }

            // ИСПРАВЛЕНИЕ: std::thread::scope стоит СНАРУЖИ, управляя временем жизни всех потоков батча
            std::thread::scope(|scope| {
                let merges_ref = &final_batch_merges;

                // Цикл внутри скоупа позволяет Rust понять, что мы берем уникальную
                // изменяемую ссылку (&mut) на КОНКРЕТНОГО воркера для каждого отдельного потока
                for worker in &mut workers {
                    scope.spawn(move || {
                        // 1. Применяем весь баг-фикс слияний без фантомных весов (weight = 0)
                        for &(id1, id2, new_id, _) in merges_ref {
                            for curr_w in 0..worker.words.len() {
                                worker.merge_tokens_inplace(curr_w, id1, id2, new_id, 0);
                            }
                        }

                        // 2. Полностью сбрасываем старую хэш-таблицу воркера
                        worker.table_keys.fill(u64::MAX);
                        worker.table_stats.fill(0);
                        worker.table_heads.fill(u32::MAX);
                        worker.next_node.fill(u32::MAX);

                        // 3. Быстрая пересборка таблицы по измененному вектору слов
                        for w_idx in 0..worker.words.len() {
                            // Вместо сохранения ссылки на весь вектор word,
                            // мы берём только его длину
                            let word_len = worker.words[w_idx].len();
                            if word_len < 2 {
                                continue;
                            }

                            let weight = worker.word_counts[w_idx] as i64;

                            for i in 0..word_len - 1 {
                                // Извлекаем ID токенов напрямую из вектора воркера.
                                // Это не создаёт долгоживущих ссылок на структуры данных!
                                let id1 = worker.words[w_idx][i] as u64;
                                let id2 = worker.words[w_idx][i + 1] as u64;
                                let pack = (id1 << 32) | id2;

                                // Теперь Rust видит, что worker полностью свободен
                                // для вызова изменяемого метода &mut self
                                worker.insert_initial(pack, weight, w_idx as u32);
                            }
                        }
                    });
                }
            }); // Все потоки гарантированно завершатся здесь перед следующей итерацией while

            if current_id % 5000 == 0 || (self.vocab_size - id_to_bytes.len() < b_size) {
                println!(
                    "[ТРЕНЕР] Собрано токенов: {} / {} ({:.2}%)",
                    id_to_bytes.len(),
                    self.vocab_size,
                    (id_to_bytes.len() as f64 / self.vocab_size as f64) * 100.0
                );
            }
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
        let mut result = String::with_capacity(bytes.len());

        for &b in bytes {
            match b {
                0x20 => result.push('Ġ'),
                0x0A => result.push('Ċ'),
                0x0D => result.push('ĉ'),
                0x09 => result.push('ĉ'),
                _ => result.push(b as char),
            }
        }
        result
    }

}
