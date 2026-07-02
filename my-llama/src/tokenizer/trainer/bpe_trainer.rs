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
        let mut counts_map: HashMap<Vec<u8>, u32> = HashMap::new();

        let text_bytes = text.as_bytes();
        let mut i = 0;
        while i < text_bytes.len() {
            let start = i;
            while i < text_bytes.len() && text_bytes[i] > 32 {
                i += 1;
            }
            if start < i {
                *counts_map.entry(text_bytes[start..i].to_vec()).or_insert(0) += 1;
            }
            i += 1;
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
            let u_slice = unsafe { TrainerUtils::byte_to_unicode_encode_fast(&b_vec, &mut static_buf) };
            vocab_json.insert(unsafe { std::str::from_utf8_unchecked(u_slice) }.to_string(), b as u32);
        }

        let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
        let chunk_size = (global_words.len() + num_threads - 1) / num_threads;
        let mut workers = Vec::with_capacity(num_threads);

        // Динамически масштабируем размер хэш-таблицы воркера на основе количества уникальных слов
        let dynamic_table_size = (chunk_size * 2).max(self.config.initial_table_size).next_power_of_two();

        for i in 0..num_threads {
            let start = (i * chunk_size).min(global_words.len());
            let end = ((i + 1) * chunk_size).min(global_words.len());
            if start < end {
                workers.push(BpeWorker::with_capacity(
                    global_words[start..end].to_vec(),
                    global_counts[start..end].to_vec(),
                    dynamic_table_size,
                ));
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
            let (mut b1, mut b2, mut bm) = ([0u8; 512], [0u8; 512], [0u8; 512]);

            for i in 0..actual_batch {
                let pack = pairs_pool[i].0;
                let (id1, id2) = ((pack >> 32) as u32, pack as u32);
                let mut merged_bytes = id_to_bytes.get(&id1).cloned().unwrap_or_default();
                merged_bytes.extend_from_slice(&id_to_bytes.get(&id2).cloned().unwrap_or_default());

                id_to_bytes.insert(current_id, merged_bytes.clone());
                let u_slice = unsafe { TrainerUtils::byte_to_unicode_encode_fast(&merged_bytes, &mut bm) };
                vocab_json.insert(unsafe { std::str::from_utf8_unchecked(u_slice) }.to_string(), current_id);

                let u1 = unsafe { std::str::from_utf8_unchecked(TrainerUtils::byte_to_unicode_encode_fast(id_to_bytes.get(&id1).unwrap(), &mut b1)) };
                let u2 = unsafe { std::str::from_utf8_unchecked(TrainerUtils::byte_to_unicode_encode_fast(id_to_bytes.get(&id2).unwrap(), &mut b2)) };
                merges.push(format!("{} {}", u1, u2));

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
        write!(writer, "{{\"version\":\"1.0\",\"model\":{{\"type\":\"BPE\",\"vocab\":{{")?;
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
