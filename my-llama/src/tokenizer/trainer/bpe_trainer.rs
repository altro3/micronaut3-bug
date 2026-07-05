use core::cmp::Ordering;
use dary_heap::OctonaryHeap;
use fxhash::FxHasher;
use std::hash::BuildHasherDefault;
use std::time::Instant;

type FxHashMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<FxHasher>>;

use crate::tokenizer::trainer::aggregator::CorpusAggregator;
use crate::tokenizer::trainer::config::TrainerConfig;
use crate::tokenizer::trainer::exporter::VocabularyExporter;
use crate::tokenizer::trainer::flat_corpus::{FlatCorpus, ProPairIndex};
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
            cyrillic_regex: r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|\p{L}+|\p{N}+|[^\s\p{L}\p{N}]+|\s+",
        }
    }

    pub fn train(&self, text_content: &str, output_json_path: &str) -> std::io::Result<()> {
        println!("[HPC ULTRA ТРЕНЕР] Шаг 1: Агрегация корпуса на потоках...");
        let global_timer = Instant::now();

        let aggregator = CorpusAggregator::new(self.cyrillic_regex, self.config.initial_table_size);
        let raw_words = aggregator.collect_unique_words(text_content.as_bytes(), self.config.num_threads, self.config.local_map_capacity);

        let mut unique_words = FxHashMap::with_capacity_and_hasher(raw_words.len(), Default::default());
        for (k, v) in raw_words {
            unique_words.insert(k, v);
        }

        println!("[HPC ULTRA ТРЕНЕР] Шаг 2: Построение плоских кэш-ориентированных индексов...");
        let (mut corpus, mut index) = FlatCorpus::build(unique_words);

        let mut current_id = self.config.start_token_id;
        let num_merges = self.vocab_size - 256;
        let mut merges_done = 0;

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

        println!("[HPC ULTRA ТРЕНЕР] Шаг 3: Запуск BPE-цикла нулевого копирования...");

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

            if merges_done % 2000 == 0 || merges_done < 5 {
                let mut b_res = id_to_bytes[job.pair.0 as usize].clone();
                b_res.extend_from_slice(&id_to_bytes[job.pair.1 as usize]);
                let clean_text = String::from_utf8_lossy(&b_res).into_owned();
                println!(
                    "[BPE LOOP] Мёрж #{:<5} | Сила сжатия: {} | Токен: '{}'",
                    merges_done,
                    job.count,
                    clean_text.escape_debug()
                );
            }

            let mut merged_bytes = id_to_bytes[job.pair.0 as usize].clone();
            merged_bytes.extend_from_slice(&id_to_bytes[job.pair.1 as usize]);
            let str_a = TrainerUtils::bytes_to_qwen_string(&id_to_bytes[job.pair.0 as usize]);
            let str_b = TrainerUtils::bytes_to_qwen_string(&id_to_bytes[job.pair.1 as usize]);
            vocab_json_output.insert(TrainerUtils::bytes_to_qwen_string(&merged_bytes), current_id);
            merges.push([str_a, str_b]);
            id_to_bytes.push(merged_bytes);

            let mut link_ptr = *index.pair_heads.get(&job.pair).unwrap_or(&-1);
            let mut spawned_pairs = Vec::with_capacity(64);

            while link_ptr != -1 {
                let w_idx = index.word_indices[link_ptr as usize];
                let n_idx = index.node_indices[link_ptr as usize];

                let node = corpus.nodes[n_idx as usize];
                if node.next != -1 && node.id == job.pair.0 && corpus.nodes[node.next as usize].id == job.pair.1 {
                    let weight = corpus.words[w_idx].weight;

                    ThreadDeltaWorker::merge_at_node(&mut corpus, n_idx, weight, current_id, |changed_pair, delta| {
                        let c = index.pair_counts.entry(changed_pair).or_insert(0);
                        *c += delta;
                        if delta > 0 {
                            spawned_pairs.push(changed_pair);
                        }
                    });
                }
                link_ptr = index.links[link_ptr as usize];
            }

            for p in spawned_pairs {
                let cnt = *index.pair_counts.get(&p).unwrap_or(&0);
                if cnt > 0 {
                    heap.push(UltraJob { count: cnt, pair: p });
                }
            }

            index.pair_counts.remove(&job.pair);

            if merges_done > 0 && merges_done % self.config.index_rebuild_interval == 0 {
                lazy_rebuild_index(&corpus, &mut index);
            }

            current_id += 1;
            merges_done += 1;
        }

        println!("[HPC ULTRA ТРЕНЕР] Шаг 4: Экспорт Qwen JSON структуры...");
        let mut std_vocab = std::collections::HashMap::with_capacity(vocab_json_output.len());
        for (k, v) in vocab_json_output {
            std_vocab.insert(k, v);
        }

        VocabularyExporter::export_qwen_json(output_json_path, &self.config, self.cyrillic_regex, std_vocab, merges, current_id)?;
        println!("[УСПЕХ] Пайплайн завершен. Время работы: {:?}", global_timer.elapsed());
        Ok(())
    }
}

fn lazy_rebuild_index(corpus: &FlatCorpus, index: &mut ProPairIndex) {
    index.pair_heads.clear();
    index.links.fill(-1);
    index.word_indices.clear();
    index.node_indices.clear();

    let mut current_link_idx = 0;

    for (w_idx, word) in corpus.words.iter().enumerate() {
        let mut curr_node_idx = word.head;
        while curr_node_idx != -1 {
            let node = corpus.nodes[curr_node_idx as usize];
            if node.next != -1 {
                let next_node = corpus.nodes[node.next as usize];
                let pair = (node.id, next_node.id);

                if index.pair_counts.contains_key(&pair) {
                    index.word_indices.push(w_idx);
                    index.node_indices.push(curr_node_idx);

                    let head = index.pair_heads.entry(pair).or_insert(-1);
                    if index.links.len() <= current_link_idx {
                        index.links.push(-1);
                    }
                    index.links[current_link_idx] = *head;
                    *head = current_link_idx as i32;

                    current_link_idx += 1;
                }
            }
            curr_node_idx = node.next;
        }
    }
}
