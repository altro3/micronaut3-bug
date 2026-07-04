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
        let mut iteration_counter = 0;

        println!("[ТРЕНЕР] Начинаю высокопроизводительный пакетный цикл слияния пар...");

        while id_to_bytes.len() < self.vocab_size {
            iteration_counter += 1;
            let mut global_pair_counts: HashMap<u64, isize, BuildTrainerHasher> = HashMap::with_hasher(BuildTrainerHasher);

            // Собираем статистику
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

            if pairs_pool.is_empty() || pairs_pool[0].1 <= 0 {
                println!("[ТРЕНЕР] Больше нет пар для слияния. Остановка.");
                break;
            }

            // РАСШИРЕННАЯ ОТЛАДКА БАТЧА
            let top_pack = pairs_pool[0].0;
            let top_id1 = (top_pack >> 32) as u32;
            let top_id2 = top_pack as u32;
            println!(
                "[ИТЕРАЦИЯ #{}] Всего уникальных пар: {}. Топ-пара: [{} + {}] с частотой: {}. Целевой ID: {}",
                iteration_counter,
                pairs_pool.len(),
                top_id1,
                top_id2,
                pairs_pool[0].1,
                current_id
            );

            // Фильтрация независимых пар
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
                println!("[ТРЕНЕР] Предупреждение: Отфильтрованный батч пуст!");
                break;
            }

            // Формируем ID и пишем в словарь
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

            let current_vocab_len = id_to_bytes.len();
            if current_vocab_len % 5000 < final_batch_merges.len() || (self.vocab_size - current_vocab_len < b_size) {
                println!(
                    "[ПРОГРЕСС] Собрано токенов: {} / {} ({:.2}%)",
                    current_vocab_len,
                    self.vocab_size,
                    (current_vocab_len as f64 / self.vocab_size as f64) * 100.0
                );
            }

            std::thread::scope(|scope| {
                let merges_ref = &final_batch_merges;

                for (w_id, worker) in workers.iter_mut().enumerate() {
                    let merges_ref = &final_batch_merges;
                    scope.spawn(move || {
                        let table_len = worker.table_keys.len();

                        // Буфер для сбора уникальных индексов слов, которые РЕАЛЬНО нужно изменить в этом батче.
                        // Используем плоский вектор для максимального кэш-хита процессора.
                        let mut words_to_modify = Vec::with_capacity(1024);

                        for &(_, _, _, old_pack) in merges_ref {
                            let base_idx = (old_pack.wrapping_mul(0x517cc1b727220a95) as usize) & worker.mask;
                            let mut real_idx = base_idx;
                            let mut probe_steps = 0;

                            loop {
                                if worker.table_keys[real_idx] == old_pack { break; }
                                if worker.table_keys[real_idx] == u64::MAX { real_idx = usize::MAX; break; }
                                probe_steps += 1;
                                if probe_steps >= table_len { real_idx = usize::MAX; break; }
                                real_idx = (real_idx + 1) & worker.mask;
                            }

                            if real_idx != usize::MAX {
                                let head = worker.table_heads[real_idx];

                                // Обнуляем статистику пары в таблице
                                worker.table_stats[real_idx] = 0;
                                worker.table_heads[real_idx] = u32::MAX;

                                // Извлекаем всю цепочку слов для этой пары ДО того, как начнем их менять
                                let mut curr_w = head as usize;
                                while curr_w != u32::MAX as usize {
                                    let next_w = worker.next_node[curr_w];

                                    // Сразу же разрываем связь в next_node, очищая глобальный граф
                                    worker.next_node[curr_w] = u32::MAX;

                                    // Сохраняем индекс слова в наш локальный буфер
                                    words_to_modify.push(curr_w);

                                    curr_w = next_w as usize;
                                }
                            }
                        }

                        // УДАЛЯЕМ ДУБЛИКАТЫ: Если слово содержало несколько пар из батча,
                        // мы оставим его строго в ОДНОМ экземпляре. Петли физически исключены!
                        words_to_modify.sort_unstable();
                        words_to_modify.dedup();

                        // Теперь последовательно и абсолютно безопасно мутируем каждое измененное слово.
                        // На Шаге 3 внутри merge_tokens_inplace они заново построят чистые, неломаные цепочки heads.
                        for curr_w in words_to_modify {
                            let weight = unsafe { *worker.word_counts.get_unchecked(curr_w) } as i64;

                            // Прогоняем по слову ВСЕ слияния текущего батча за один раз!
                            for &(id1, id2, new_id, _) in merges_ref {
                                worker.merge_tokens_inplace(curr_w, id1, id2, new_id, weight);
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
