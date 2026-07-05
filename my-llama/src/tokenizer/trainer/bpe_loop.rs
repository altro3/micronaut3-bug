use crate::tokenizer::trainer::config::TrainerConfig;
use crate::tokenizer::trainer::job::UltraJob;
use crate::tokenizer::trainer::position_index::{IsolatedWord, PositionIndex};
use crate::tokenizer::trainer::telemetry::BpeTelemetry;
use crate::tokenizer::trainer::worker::ThreadDeltaWorker;
use dary_heap::OctonaryHeap;
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
        words: &mut Vec<IsolatedWord>,
        index: &mut PositionIndex,
        heap: &mut OctonaryHeap<UltraJob>,
        id_to_bytes: &mut Vec<Vec<u8>>,
    ) -> Vec<(u32, u32)> {
        let num_merges = self.config.vocab_size - 256;
        let mut merges_done = 0;
        let mut current_id = self.config.start_token_id;
        let mut raw_merges = Vec::with_capacity(num_merges);

        println!("\n================== [ЗАПУСК КАНОНИЧЕСКОГО BPE-ЦИКЛА В СТИЛЕ HUGGING FACE] ==================");
        let loop_start = Instant::now();

        while merges_done < num_merges {
            let Some(job) = heap.pop() else {
                println!("[КРИТИЧЕСКИЙ СБОЙ] Приоритетная очередь внезапно опустела на итерации #{}!", merges_done);
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
                println!("[ИНФО] Сила сжатия упала до нуля. Словарь собран досрочно.");
                break;
            }

            BpeTelemetry::log_progress(merges_done, num_merges, current_id, &job, id_to_bytes, index, heap, loop_start);

            let mut merged_bytes = id_to_bytes[job.pair.0 as usize].clone();
            merged_bytes.extend_from_slice(&id_to_bytes[job.pair.1 as usize]);
            raw_merges.push((job.pair.0, job.pair.1));

            if (current_id as usize) >= id_to_bytes.len() {
                id_to_bytes.resize(current_id as usize + 1, vec![]);
            }
            id_to_bytes[current_id as usize] = merged_bytes;

            self.process_word_merges_flat(words, index, heap, job.pair, current_id, merges_done);

            index.pair_counts.remove(&job.pair);
            current_id += 1;
            merges_done += 1;
        }

        println!("==============================================================================");
        println!("[ТРЕНЕР] Основной цикл BPE завершен за: {:?}", loop_start.elapsed());
        raw_merges
    }

    fn process_word_merges_flat(
        &self,
        words: &mut Vec<IsolatedWord>,
        index: &mut PositionIndex,
        heap: &mut OctonaryHeap<UltraJob>,
        target_pair: (u32, u32),
        new_id: u32,
        merges_done: usize,
    ) {
        let mut words_modified = 0;
        let mut total_lazy_pushes = 0;

        for w_idx in 0..words.len() {
            let word = &mut words[w_idx];

            let added_pairs = ThreadDeltaWorker::merge_in_word(word, w_idx as u32, target_pair, new_id, index);

            if !added_pairs.is_empty() {
                words_modified += 1;
                for p in added_pairs {
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
                "      ├── [ДЕТАЛИ МЁРЖА] Модифицировано слов: {} | Ленивых пушей в кучу: {}",
                words_modified, total_lazy_pushes
            );
        }
    }
}
