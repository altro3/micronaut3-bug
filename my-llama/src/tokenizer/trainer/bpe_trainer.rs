use crate::tokenizer::trainer::utils::TrainerUtils;
use crate::tokenizer::trainer::{BpeWorker, BuildTrainerHasher};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
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
}

impl BpeTrainer {
    pub fn new(vocab_size: usize, config: TrainerConfig) -> Self {
        Self { vocab_size, config }
    }

    pub fn train(&self, text: &str, output_json_path: &str) -> std::io::Result<()> {
        let timer = Instant::now();
        let text_bytes = text.as_bytes();
        let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);

        let chunk_size = (text_bytes.len() + num_threads - 1) / num_threads;
        let mut global_maps: Vec<Box<HashMap<Vec<u8>, u32>>> = vec![Box::new(HashMap::new()); num_threads];

        std::thread::scope(|scope| {
            for (t_idx, local_map) in global_maps.iter_mut().enumerate() {
                scope.spawn(move || {
                    let start_pos = (t_idx * chunk_size).min(text_bytes.len());
                    let mut end_pos = ((t_idx + 1) * chunk_size).min(text_bytes.len());
                    while end_pos < text_bytes.len() && text_bytes[end_pos] > 32 {
                        end_pos += 1;
                    }
                    let local_chunk = &text_bytes[start_pos..end_pos];
                    let mut i = 0;
                    while i < local_chunk.len() {
                        let start = i;
                        while i < local_chunk.len() && local_chunk[i] > 32 {
                            i += 1;
                        }
                        if start < i {
                            *local_map.entry(local_chunk[start..i].to_vec()).or_insert(0) += 1;
                        }
                        i += 1;
                    }
                });
            }
        });

        let mut counts_map = HashMap::with_capacity(65536);
        for local_map in global_maps {
            for (k, v) in *local_map {
                counts_map.insert(k, v);
            }
        }

        let (mut global_words, mut global_counts) = (Vec::new(), Vec::new());
        for (w_bytes, count) in counts_map {
            global_words.push(w_bytes.iter().map(|&b| b as u32).collect::<Vec<u32>>());
            global_counts.push(count);
        }

        let (mut id_to_bytes, mut vocab_json) = (HashMap::with_hasher(BuildTrainerHasher), HashMap::with_hasher(BuildTrainerHasher));
        let mut merges = Vec::with_capacity(self.vocab_size);
        let mut static_buf = [0u8; 512];

        for b in 0..=255 {
            let b_vec = vec![b];
            id_to_bytes.insert(b as u32, b_vec.clone());
            let u_slice = TrainerUtils::byte_to_unicode_encode_fast(&b_vec, &mut static_buf);
            vocab_json.insert(unsafe { std::str::from_utf8_unchecked(u_slice) }.to_string(), b as u32);
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

        while id_to_bytes.len() < self.vocab_size {
            let mut global_pair_counts: HashMap<u64, isize, BuildTrainerHasher> = HashMap::with_hasher(BuildTrainerHasher);
            for worker in &workers {
                let local_pairs = worker
                    .table_keys
                    .iter()
                    .enumerate()
                    .filter(|&(idx, &key)| key != u64::MAX && worker.table_stats[idx] > 0)
                    .take(b_size * 2);
                for (idx, &pack) in local_pairs {
                    *global_pair_counts.entry(pack).or_insert(0) += worker.table_stats[idx] as isize;
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
                merges.push(format!("{} {}", id1, id2));
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
                                    unsafe {
                                        worker.merge_tokens_inplace(curr_w, id1, id2, new_id, weight);
                                    }
                                    curr_w = worker.next_node[curr_w] as usize;
                                }
                            }
                        }
                    });
                }
            });
        }

        let mut writer = BufWriter::with_capacity(self.config.io_buffer_size, File::create(output_json_path)?);
        write!(writer, "{{\"\x76ersion\":\"1.0\",\"model\":{{\"type\":\"BPE\",\"vocab\":{{")?;
        for (i, (k, v)) in vocab_json.iter().enumerate() {
            write!(writer, "\"{}\":{}", k, v)?;
            if i < vocab_json.len() - 1 {
                write!(writer, ",")?;
            }
        }
        write!(writer, "}},\"merges\":[")?;
        for (i, m) in merges.iter().enumerate() {
            write!(writer, "\"{}\"", m)?;
            if i < merges.len() - 1 {
                write!(writer, ",")?;
            }
        }
        write!(writer, "]}}}}")?;
        writer.flush()?;
        println!("[Trainer] Обучение успешно завершено за: {:?}", timer.elapsed());
        Ok(())
    }
}
