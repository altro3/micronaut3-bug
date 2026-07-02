use super::simd_splitter::SimdSplitter;
use rustc_hash::{FxHashMap, FxHashSet};
use serde_json::json;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

fn byte_to_unicode_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len());
    for &b in bytes {
        let u = b as u32;
        let c = match b {
            0..=32 | 127..=159 => char::from_u32(u).unwrap(),
            33..=126 => char::from_u32(u).unwrap(),
            160..=191 => char::from_u32(u + 0x0100 - 33).unwrap(),
            192..=222 => char::from_u32(u + 0x0120 - 127).unwrap(),
            223..=255 => char::from_u32(u + 0x0180 - 223).unwrap(),
        };
        s.push(c);
    }
    s
}

struct BpeWorker {
    pub words: Vec<Vec<u32>>,
    pub word_counts: Vec<usize>,
    pub pair_stats: FxHashMap<u64, isize>,
    pub pair_to_words: FxHashMap<u64, FxHashSet<usize>>,
}

impl BpeWorker {
    pub fn new(words: Vec<Vec<u32>>, word_counts: Vec<usize>) -> Self {
        let mut pair_stats = FxHashMap::default();
        let mut pair_to_words: FxHashMap<u64, FxHashSet<usize>> = FxHashMap::default();

        for (w_idx, word) in words.iter().enumerate() {
            if word.len() < 2 {
                continue;
            }
            let weight = word_counts[w_idx] as isize;
            for i in 0..word.len() - 1 {
                let pack = ((word[i] as u64) << 32) | (word[i + 1] as u64);
                *pair_stats.entry(pack).or_default() += weight;
                pair_to_words.entry(pack).or_default().insert(w_idx);
            }
        }

        Self {
            words,
            word_counts,
            pair_stats,
            pair_to_words,
        }
    }
}

pub struct BpeTrainer {
    vocab_size: usize,
}

impl BpeTrainer {
    pub fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }

    pub fn train(&self, text: &str, output_json_path: &str) -> std::io::Result<()> {
        let timer = Instant::now();
        println!("[Trainer MapReduce] Шаг 1: Пре-токенизация 800МБ датасета через SIMD...");

        let mut global_words: Vec<Vec<u32>> = Vec::new();
        let mut global_word_counts: Vec<usize> = Vec::new();
        let mut counts_map: FxHashMap<Vec<u8>, usize> = FxHashMap::default();

        SimdSplitter::split(text, |chunk| {
            if !chunk.is_empty() {
                *counts_map.entry(chunk.to_vec()).or_insert(0) += 1;
            }
        });

        for (word_bytes, count) in counts_map {
            let word_ids: Vec<u32> = word_bytes.iter().map(|&b| b as u32).collect();
            global_words.push(word_ids);
            global_word_counts.push(count);
        }

        let mut id_to_bytes: FxHashMap<u32, Vec<u8>> = FxHashMap::default();
        let mut vocab_json: FxHashMap<String, u32> = FxHashMap::default();

        for b in 0..=255 {
            let b_vec = vec![b];
            id_to_bytes.insert(b as u32, b_vec.clone());
            vocab_json.insert(byte_to_unicode_encode(&b_vec), b as u32);
        }

        println!("[Trainer MapReduce] Шаг 2: Нарезка и изолированная инициализация воркеров по ядрам CPU...");
        let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let chunk_size = (global_words.len() + num_threads - 1) / num_threads;

        let mut workers = Vec::with_capacity(num_threads);

        // Разделяем глобальные векторы на куски для независимых потоков
        let mut words_chunks: Vec<Vec<Vec<u32>>> = vec![Vec::new(); num_threads];
        let mut counts_chunks: Vec<Vec<usize>> = vec![Vec::new(); num_threads];

        let mut current_thread = 0;
        for (idx, word) in global_words.into_iter().enumerate() {
            if idx > 0 && idx % chunk_size == 0 && current_thread < num_threads - 1 {
                current_thread += 1;
            }
            words_chunks[current_thread].push(word);
            counts_chunks[current_thread].push(global_word_counts[idx]);
        }

        // Параллельно инициализируем внутренние индексы воркеров без взаимных блокировок
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(num_threads);
            for (w_chunk, c_chunk) in words_chunks.into_iter().zip(counts_chunks.into_iter()) {
                handles.push(scope.spawn(move || BpeWorker::new(w_chunk, c_chunk)));
            }
            for handle in handles {
                workers.push(handle.join().unwrap());
            }
        });

        let mut merges: Vec<String> = Vec::new();
        let mut current_id = 256u32;

        println!("[Trainer MapReduce] Шаг 3: Запуск итерационного Reduce-цикла слияния на 200 000 токенов...");
        let batch_size = 128;

        while id_to_bytes.len() < self.vocab_size {
            let mut global_pair_counts: FxHashMap<u64, isize> = FxHashMap::default();

            for worker in &workers {
                let local_pairs: Vec<(&u64, &isize)> = worker.pair_stats.iter().filter(|&(_, &count)| count > 0).take(batch_size * 2).collect();

                for (&pack, &count) in local_pairs {
                    *global_pair_counts.entry(pack).or_insert(0) += count;
                }
            }

            let mut pairs_pool: Vec<(u64, isize)> = global_pair_counts.into_iter().collect();
            pairs_pool.sort_unstable_by_key(|&(_, count)| -count);

            if pairs_pool.is_empty() || pairs_pool[0].1 <= 0 {
                println!("[Trainer MapReduce] Текст полностью исчерпан.");
                break;
            }

            let actual_batch = batch_size.min(pairs_pool.len()).min(self.vocab_size - id_to_bytes.len());

            let mut batch_merges = Vec::with_capacity(actual_batch);
            for i in 0..actual_batch {
                let pack = pairs_pool[i].0;
                let id1 = (pack >> 32) as u32;
                let id2 = pack as u32;

                let bytes1 = id_to_bytes.get(&id1).cloned().unwrap_or_default();
                let bytes2 = id_to_bytes.get(&id2).cloned().unwrap_or_default();
                let mut merged_bytes = bytes1.clone();
                merged_bytes.extend_from_slice(&bytes2);

                id_to_bytes.insert(current_id, merged_bytes.clone());

                let u1 = byte_to_unicode_encode(&bytes1);
                let u2 = byte_to_unicode_encode(&bytes2);
                let merged_unicode = byte_to_unicode_encode(&merged_bytes);

                vocab_json.insert(merged_unicode, current_id);
                merges.push(format!("{} {}", u1, u2));

                batch_merges.push((id1, id2, current_id, pack));
                current_id += 1;
            }

            // Запускаем воркеры сжимать локальные тексты пачкой
            std::thread::scope(|scope| {
                for worker in &mut workers {
                    let merges_ref = &batch_merges;
                    scope.spawn(move || {
                        for &(id1, id2, new_id, old_pack) in merges_ref {
                            worker.pair_stats.insert(old_pack, 0);

                            if let Some(affected_words) = worker.pair_to_words.remove(&old_pack) {
                                for w_idx in affected_words {
                                    let word = unsafe { worker.words.get_unchecked_mut(w_idx) };
                                    let weight = unsafe { *worker.word_counts.get_unchecked(w_idx) } as isize;

                                    if word.len() < 2 {
                                        continue;
                                    }

                                    for i in 0..word.len() - 1 {
                                        let old_p = ((word[i] as u64) << 32) | (word[i + 1] as u64);
                                        if let Some(stat) = worker.pair_stats.get_mut(&old_p) {
                                            *stat -= weight;
                                        }
                                    }

                                    let mut i = 0;
                                    let mut new_word = Vec::with_capacity(word.len());
                                    while i < word.len() {
                                        if i < word.len() - 1 && word[i] == id1 && word[i + 1] == id2 {
                                            new_word.push(new_id);
                                            i += 2;
                                        } else {
                                            new_word.push(word[i]);
                                            i += 1;
                                        }
                                    }
                                    *word = new_word;

                                    if word.len() >= 2 {
                                        for i in 0..word.len() - 1 {
                                            let new_p = ((word[i] as u64) << 32) | (word[i + 1] as u64);
                                            *worker.pair_stats.entry(new_p).or_insert(0) += weight;
                                            worker.pair_to_words.entry(new_p).or_default().insert(w_idx);
                                        }
                                    }
                                }
                            }
                        }
                    });
                }
            });

            if id_to_bytes.len() % 10000 == 0 || id_to_bytes.len() == self.vocab_size {
                println!(
                    "|-> [MapReduce 100% CPU] Токенов сгенерировано: {} / {} [{:.2?}]",
                    id_to_bytes.len(),
                    self.vocab_size,
                    timer.elapsed()
                );
            }
        }

        println!("[Trainer MapReduce] Сборка и сохранение финального JSON-словаря...");
        let json_data = json!({
            "version": "1.0",
            "added_tokens": [
                { "id": current_id, "content": "<|endoftext|>", "special": true }
            ],
            "model": {
                "type": "BPE",
                "vocab": vocab_json,
                "merges": merges
            }
        });

        let mut file = File::create(output_json_path)?;
        file.write_all(serde_json::to_string_pretty(&json_data)?.as_bytes())?;
        println!("[Trainer MapReduce] Обучение успешно завершено! Итоговое время: {:?}", timer.elapsed());

        Ok(())
    }
}
