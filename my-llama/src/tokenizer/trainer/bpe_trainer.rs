use dary_heap::OctonaryHeap;
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use core::cmp::Ordering;

use crate::tokenizer::trainer::aggregator::CorpusAggregator;
use crate::tokenizer::trainer::config::TrainerConfig;
use crate::tokenizer::trainer::exporter::VocabularyExporter;
use crate::tokenizer::trainer::flat_corpus::FlatCorpus;
use crate::tokenizer::trainer::utils::TrainerUtils;
use crate::tokenizer::trainer::worker::ThreadDeltaWorker;

#[derive(Debug, Eq)]
pub struct MergeJob {
    pub count: i64,
    pub pair: (u32, u32),
    pub word_indices: HashSet<usize>,
}

impl MergeJob {
    pub fn heap_key(&self) -> (i64, (u32, u32)) {
        (self.count, self.pair)
    }
}

impl PartialEq for MergeJob {
    fn eq(&self, other: &Self) -> bool {
        self.heap_key() == other.heap_key()
    }
}

impl PartialOrd for MergeJob {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeJob {
    fn cmp(&self, other: &Self) -> Ordering {
        self.heap_key().cmp(&other.heap_key())
    }
}

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
        println!("[HPC ТРЕНЕР] Шаг 1: Агрегация корпуса на потоках...");
        let global_timer = Instant::now();

        let aggregator = CorpusAggregator::new(self.cyrillic_regex, self.config.initial_table_size);
        let unique_words = aggregator.collect_unique_words(
            text_content.as_bytes(),
            self.config.num_threads,
            self.config.local_map_capacity,
        );

        println!("[HPC ТРЕНЕР] Шаг 2: Построение кучи и индексов...");
        let (mut corpus, mut pair_index_data) = FlatCorpus::build(unique_words);

        let mut current_id = self.config.start_token_id;
        let num_merges = self.vocab_size - 256;
        let mut merges_done = 0;

        let mut id_to_bytes: Vec<Vec<u8>> = (0..256).map(|b| vec![b as u8]).collect();
        id_to_bytes.reserve(self.vocab_size);

        let mut merges: Vec<[String; 2]> = Vec::with_capacity(self.vocab_size);
        let mut vocab_json_output = HashMap::with_capacity(self.vocab_size);
        for b in 0..256 {
            let b_vec = vec![b as u8];
            vocab_json_output.insert(TrainerUtils::bytes_to_qwen_string(&b_vec), b as u32);
        }

        let mut pair_counts = pair_index_data.pair_counts;
        let table_pair_index = pair_index_data.pair_index;

        // Инициализируем оригинальную 8-арную кучу wordchipper
        let mut heap = OctonaryHeap::with_capacity(pair_counts.len());
        for (pair, word_indices) in table_pair_index.into_iter() {
            let count = *pair_counts.get(&pair).unwrap_or(&0);
            if count > 0 {
                heap.push(MergeJob { pair, count, word_indices });
            }
        }

        println!("[HPC ТРЕНЕР] Шаг 3: Выполнение реактивного BPE цикла...");

        while merges_done < num_merges {
            let Some(mut job) = heap.pop() else { break; };

            // ЛЕНИВОЕ ОБНОВЛЕНИЕ КУЧИ (Оригинальный паттерн wordchipper)
            let current_count = *pair_counts.get(&job.pair).unwrap_or(&0);
            if job.count != current_count {
                job.count = current_count;
                if job.count > 0 {
                    heap.push(job);
                }
                continue;
            }

            if job.count == 0 { break; }

            // Выводим логи логарифмически для верификации
            if merges_done % 1000 == 0 || merges_done < 10 {
                let mut b_res = id_to_bytes[job.pair.0 as usize].clone();
                b_res.extend_from_slice(&id_to_bytes[job.pair.1 as usize]);
                let clean_text = String::from_utf8_lossy(&b_res).into_owned();
                let progress = (current_id as f64 / self.vocab_size as f64) * 100.0;

                println!(
                    "[РЕАКТИВНЫЙ BPE] Шаг: {:<6} | Сжатие: {:.2}% | Мёрж: '{}' (частота: {})",
                    merges_done, progress, clean_text.escape_debug(), job.count
                );
            }

            // Регистрируем токен
            let mut merged_bytes = id_to_bytes[job.pair.0 as usize].clone();
            merged_bytes.extend_from_slice(&id_to_bytes[job.pair.1 as usize]);
            let str_a = TrainerUtils::bytes_to_qwen_string(&id_to_bytes[job.pair.0 as usize]);
            let str_b = TrainerUtils::bytes_to_qwen_string(&id_to_bytes[job.pair.1 as usize]);
            vocab_json_output.insert(TrainerUtils::bytes_to_qwen_string(&merged_bytes), current_id);
            merges.push([str_a, str_b]);
            id_to_bytes.push(merged_bytes);

            // Карта для отслеживания новых пар, рожденных на этом шаге
            let mut new_token_pair_map: HashMap<(u32, u32), HashSet<usize>> = HashMap::with_capacity(32);

            // Мёржим строго в тех словах, где эта пара существует! O(Очаги), а не O(Весь корпус)
            for &word_idx in &job.word_indices {
                let word = &mut corpus.words[word_idx];

                ThreadDeltaWorker::merge_pair_with_callback(word, job.pair, current_id, |pair, delta| {
                    let count_ref = pair_counts.entry(pair).or_insert(0);
                    *count_ref += delta;

                    if delta > 0 {
                        // Регистрируем положение новой родившейся пары для кучи
                        new_token_pair_map.entry(pair).or_insert_with(HashSet::new).insert(word_idx);
                    }
                });
            }

            // Добавляем новые родившиеся пары в кучу
            for (pair, word_indices) in new_token_pair_map {
                let count = *pair_counts.get(&pair).unwrap_or(&0);
                if count > 0 {
                    heap.push(MergeJob { pair, count, word_indices });
                }
            }

            pair_counts.remove(&job.pair);
            current_id += 1;
            merges_done += 1;
        }

        println!("[ЭКСПОРТ] Запись JSON структуры на диск...");
        VocabularyExporter::export_qwen_json(output_json_path, &self.config, self.cyrillic_regex, vocab_json_output, merges, current_id)?;
        println!("[УСПЕХ] Пайплайн полностью завершен за: {:?}", global_timer.elapsed());
        Ok(())
    }
}
