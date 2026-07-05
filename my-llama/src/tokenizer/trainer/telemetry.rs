use crate::tokenizer::trainer::job::UltraJob;
use crate::tokenizer::trainer::position_index::PositionIndex;
use dary_heap::OctonaryHeap;
use std::time::Instant;

pub struct BpeTelemetry;

impl BpeTelemetry {
    pub fn log_progress(
        merges_done: usize,
        num_merges: usize,
        current_id: u32,
        job: &UltraJob,
        id_to_bytes: &[Vec<u8>],
        index: &PositionIndex,
        heap: &OctonaryHeap<UltraJob>,
        loop_start: Instant,
    ) {
        if merges_done % 1000 == 0 || merges_done < 20 {
            let mut real_bytes = id_to_bytes[job.pair.0 as usize].clone();
            real_bytes.extend_from_slice(&id_to_bytes[job.pair.1 as usize]);
            let clean_text = String::from_utf8_lossy(&real_bytes).into_owned();
            let pct = (merges_done as f64 / num_merges as f64) * 100.0;

            let elapsed_loop = loop_start.elapsed().as_secs_f64();
            let speed_merges = if elapsed_loop > 0.0 { merges_done as f64 / elapsed_loop } else { 0.0 };

            println!(
                "[BPE LOOP] Итерация #{:<5} ({:04.1}%) | Сила: {:<10} | Токен ID: {:<5} | Слияний/сек: {:.0} | Текст: '{}'",
                merges_done,
                pct,
                job.count,
                current_id,
                speed_merges,
                clean_text.escape_debug()
            );

            if merges_done % 1000 == 0 && merges_done > 0 {
                println!(
                    "  [МОНИТОРИНГ ОЗУ] Активных пар: {} | Записей в индексе: {} | Размер кучи: {}",
                    index.pair_counts.len(),
                    index.pair_to_words.len(),
                    heap.len()
                );
            }
        }
    }
}
