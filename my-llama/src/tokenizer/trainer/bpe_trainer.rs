use core::cmp::Ordering;
use dary_heap::OctonaryHeap;
use fxhash::FxHasher;
use std::hash::BuildHasherDefault;

type FxHashMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<FxHasher>>;

use crate::tokenizer::trainer::aggregator::CorpusAggregator;
use crate::tokenizer::trainer::config::TrainerConfig;
use crate::tokenizer::trainer::exporter::VocabularyExporter;
use crate::tokenizer::trainer::flat_corpus::FlatCorpus;
use crate::tokenizer::trainer::utils::TrainerUtils;
use crate::tokenizer::trainer::worker::ThreadDeltaWorker;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct UltraJob {
    pub count: i64,
    pub pair: (u32, u32),
}

impl Ord for UltraJob {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        self.count.cmp(&other.count).then_with(|| self.pair.cmp(&other.pair))
    }
}

impl PartialOrd for UltraJob {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
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
        let aggregator = CorpusAggregator::new(self.cyrillic_regex, self.config.initial_table_size);
        let raw_words = aggregator.collect_unique_words(text_content.as_bytes(), self.config.num_threads, self.config.local_map_capacity);

        let mut unique_words = FxHashMap::with_capacity_and_hasher(raw_words.len(), Default::default());
        for (k, v) in raw_words {
            unique_words.insert(k, v);
        }

        let (mut corpus, mut index) = FlatCorpus::build(unique_words);

        let mut current_id = self.config.start_token_id;
        let num_merges = self.vocab_size - 256;
        let mut merges_done = 0;

        // В id_to_bytes лежат чистые, сырые Qwen-байты (0..256), как они прилетели из агрегатора
        let mut id_to_bytes: Vec<Vec<u8>> = (0..256).map(|b| vec![b as u8]).collect();
        id_to_bytes.reserve(self.vocab_size);

        let mut merges: Vec<[String; 2]> = Vec::with_capacity(self.vocab_size);
        let mut vocab_json_output = FxHashMap::with_capacity_and_hasher(self.vocab_size, Default::default());

        for b in 0..256 {
            let b_vec = vec![b as u8];
            vocab_json_output.insert(TrainerUtils::bytes_to_qwen_string(&b_vec), b as u32);
        }

        let mut heap = OctonaryHeap::with_capacity(index.pair_counts.len());
        for (&pair, &count) in &index.pair_counts {
            if count > 0 {
                heap.push(UltraJob { count, pair });
            }
        }

        let mut spawned_positions: FxHashMap<(u32, u32), Vec<i32>> = FxHashMap::default();

        println!("[HPC ULTRA ТРЕНЕР] Запуск стабильного BPE-цикла...");

        while merges_done < num_merges {
            let Some(job) = heap.pop() else {
                break;
            };

            let current_count = *index.pair_counts.get(&job.pair).unwrap_or(&0);
            if job.count != current_count {
                if current_count > 0 {
                    heap.push(UltraJob {
                        count: current_count,
                        pair: job.pair,
                    });
                }
                continue;
            }
            if job.count <= 0 {
                break;
            }

            // ИДЕАЛЬНОЕ И БЕЗОПАСНОЕ ДЕКОДИРОВАНИЕ СТРОГО ДЛЯ ВЫВОДА В ЛОГ:
            if merges_done % 2000 == 0 || merges_done < 10 {
                let mut b_res = id_to_bytes[job.pair.0 as usize].clone();
                b_res.extend_from_slice(&id_to_bytes[job.pair.1 as usize]);

                // Переводим накопленные латинские байты в Qwen-строку (это всегда 100% валидный ASCII/Latin)
                if let Ok(qwen_str) = String::from_utf8(b_res) {
                    // Восстанавливаем оригинальные сырые байты кириллицы
                    let real_bytes = TrainerUtils::qwen_string_to_bytes(&qwen_str);
                    // Выводим чистый русский текст без кракозябр и знаков ошибок
                    let clean_text = String::from_utf8_lossy(&real_bytes).into_owned();

                    println!(
                        "[BPE LOOP] Мёрж #{:<5} | Сила сжатия: {:<10} | Токен: '{}'",
                        merges_done,
                        job.count,
                        clean_text.escape_debug()
                    );
                }
            }

            let mut merged_bytes = id_to_bytes[job.pair.0 as usize].clone();
            merged_bytes.extend_from_slice(&id_to_bytes[job.pair.1 as usize]);

            // Формируем чистые Qwen-строки для финального JSON экспорта
            let str_a = String::from_utf8(id_to_bytes[job.pair.0 as usize].clone()).unwrap();
            let str_b = String::from_utf8(id_to_bytes[job.pair.1 as usize].clone()).unwrap();
            let str_merged = String::from_utf8(merged_bytes.clone()).unwrap();

            vocab_json_output.insert(str_merged, current_id);
            merges.push([str_a, str_b]);
            id_to_bytes.push(merged_bytes);

            if let Some(&(start, count)) = index.pair_slices.get(&job.pair) {
                let slice = &index.positions[start..start + count];
                for &n_idx in slice {
                    if let Some(&node) = corpus.nodes.get(n_idx as usize) {
                        if node.next != -1 && node.id == job.pair.0 && corpus.nodes[node.next as usize].id == job.pair.1 {
                            let w_idx = node.word_idx as usize;
                            let weight = corpus.words[w_idx].weight;

                            ThreadDeltaWorker::merge_at_node(&mut corpus, n_idx, weight, current_id, |changed_pair, delta| {
                                *index.pair_counts.entry(changed_pair).or_insert(0) += delta;
                                if delta > 0 {
                                    spawned_positions.entry(changed_pair).or_insert_with(Vec::new).push(n_idx);
                                }
                            });
                        }
                    }
                }
            }

            if let Some(dyn_slice) = spawned_positions.remove(&job.pair) {
                for n_idx in dyn_slice {
                    if let Some(&node) = corpus.nodes.get(n_idx as usize) {
                        if node.next != -1 && node.id == job.pair.0 && corpus.nodes[node.next as usize].id == job.pair.1 {
                            let w_idx = node.word_idx as usize;
                            let weight = corpus.words[w_idx].weight;

                            ThreadDeltaWorker::merge_at_node(&mut corpus, n_idx, weight, current_id, |changed_pair, delta| {
                                *index.pair_counts.entry(changed_pair).or_insert(0) += delta;
                                if delta > 0 {
                                    spawned_positions.entry(changed_pair).or_insert_with(Vec::new).push(n_idx);
                                }
                            });
                        }
                    }
                }
            }

            for &p in spawned_positions.keys() {
                let cnt = *index.pair_counts.get(&p).unwrap_or(&0);
                if cnt > 0 {
                    heap.push(UltraJob { count: cnt, pair: p });
                }
            }

            index.pair_counts.remove(&job.pair);
            current_id += 1;
            merges_done += 1;
        }

        let mut std_vocab = std::collections::HashMap::with_capacity(vocab_json_output.len());
        for (k, v) in vocab_json_output {
            std_vocab.insert(k, v);
        }

        VocabularyExporter::export_qwen_json(output_json_path, &self.config, self.cyrillic_regex, std_vocab, merges, current_id)?;
        Ok(())
    }
}
