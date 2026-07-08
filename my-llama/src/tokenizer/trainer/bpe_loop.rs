use crate::tokenizer::trainer::config::TrainerConfig;
use crate::tokenizer::trainer::job::UltraJob;
use crate::tokenizer::trainer::position_index::{IsolatedWord, PositionIndex};
use crate::tokenizer::trainer::telemetry::BpeTelemetry;
use crate::tokenizer::trainer::worker::ThreadDeltaWorker;
use dary_heap::OctonaryHeap;
use std::ptr::copy_nonoverlapping;
use std::time::Instant;

pub struct BpeLoopRunner<'a> {
    config: &'a TrainerConfig,
}

impl<'a> BpeLoopRunner<'a> {
    pub fn new(config: &'a TrainerConfig) -> Self {
        Self { config }
    }

    pub fn run(
        &self,
        words: &mut [IsolatedWord],
        index: &mut PositionIndex,
        heap: &mut OctonaryHeap<UltraJob>,
        vocab_bytes_flat: &mut Vec<u8>,
        vocab_offsets_flat: &mut [u64],
    ) -> Vec<(u32, u32)> {
        let num_merges = self.config.vocab_size - 256;
        let mut merges_done = 0;
        let mut current_id = self.config.start_token_id;
        let mut raw_merges = Vec::with_capacity(num_merges);

        println!("\n================== [ЗАПУСК ОПТИМИЗИРОВАННОГО ВЕКТОРНОГО BPE-ЦИКЛА] ==================");
        let loop_start = Instant::now();

        while merges_done < num_merges {
            let Some(job) = heap.pop() else {
                println!("[КРИТИЧЕСКИЙ СБОЙ] Приоритетная очередь внезапно опустела на итерации #{}!", merges_done);
                break;
            };

            let current_count = *index.pair_counts.get(&job.pair).unwrap_or(&0);
            if job.count != current_count {
                if current_count > 0 {
                    heap.push(UltraJob { count: current_count, pair: job.pair });
                }
                continue;
            }

            if job.count <= 0 {
                println!("[ИНФО] Сила сжатия упала до нуля. Словарь собран досрочно.");
                break;
            }

            let packed_a = vocab_offsets_flat[job.pair.0 as usize];
            let packed_b = vocab_offsets_flat[job.pair.1 as usize];

            let len_a = (packed_a & 0xFFFFFFFF) as usize;
            let len_b = (packed_b & 0xFFFFFFFF) as usize;
            let total_len = len_a + len_b;

            if total_len > self.config.max_token_length {
                if let Some(cnt) = index.pair_counts.get_mut(&job.pair) {
                    *cnt = 0;
                }
                continue;
            }

            BpeTelemetry::log_progress(
                merges_done,
                num_merges,
                current_id,
                &job,
                vocab_bytes_flat,
                vocab_offsets_flat,
                index,
                heap,
                loop_start,
            );

            let offset_a = (packed_a >> 32) as usize;
            let offset_b = (packed_b >> 32) as usize;

            let new_offset = vocab_bytes_flat.len();
            vocab_bytes_flat.resize(new_offset + total_len, 0);

            unsafe {
                let src_ptr = vocab_bytes_flat.as_ptr();
                let dst_ptr = vocab_bytes_flat.as_mut_ptr().add(new_offset);

                copy_nonoverlapping(src_ptr.add(offset_a), dst_ptr, len_a);
                copy_nonoverlapping(src_ptr.add(offset_b), dst_ptr.add(len_a), len_b);
            }

            vocab_offsets_flat[current_id as usize] = ((new_offset as u64) << 32) | (total_len as u64);
            raw_merges.push((job.pair.0, job.pair.1));

            self.process_word_merges(words, index, heap, job.pair, current_id, merges_done);

            if let Some(cnt) = index.pair_counts.get_mut(&job.pair) {
                *cnt = 0;
            }

            current_id += 1;
            merges_done += 1;
        }

        println!("==============================================================================");
        raw_merges
    }

    fn process_word_merges(
        &self,
        words: &mut [IsolatedWord],
        index: &mut PositionIndex,
        heap: &mut OctonaryHeap<UltraJob>,
        target_pair: (u32, u32),
        new_id: u32,
        merges_done: usize,
    ) {
        let mut words_modified = 0;
        let mut total_lazy_pushes = 0;

        if let Some(word_ids) = index.pair_to_words.remove(&target_pair) {
            words_modified = word_ids.len();
            let mut added_pairs_scratch = Vec::with_capacity(16);

            for w_idx in word_ids {
                let word = &mut words[w_idx as usize];

                ThreadDeltaWorker::merge_in_word(word, w_idx, target_pair, new_id, index, &mut added_pairs_scratch);

                for &p in &added_pairs_scratch {
                    let words_vec = index.pair_to_words.entry(p).or_default();
                    if words_vec.last() != Some(&w_idx) {
                        words_vec.push(w_idx);
                    }

                    let cnt = *index.pair_counts.get(&p).unwrap_or(&0);
                    if cnt > 0 {
                        heap.push(UltraJob { count: cnt, pair: p });
                        total_lazy_pushes += 1;
                    }
                }
            }
        }

        if merges_done < 20 {
            println!(
                "      ├── [ДЕТАЛИ МЁРЖА] Затронуто слов: {} | Ленивых пушей в кучу: {}",
                words_modified, total_lazy_pushes
            );
        }
    }
}
